#pragma once

// 图和 KV 由草稿实例拥有；目标验证成功前，草稿位置不会推进。
struct DSparkRuntime {
 QNN_INTERFACE_VER_TYPE& api;
 std::function<void(CachedGraph&)> free_graph;
 std::vector<CachedGraph> graphs;
 unsigned capacity,hidden,dim,kv_heads,layers,rows;
 std::vector<unsigned> target_layers;
 std::vector<std::vector<uint16_t>> keys,values;
 std::vector<uint16_t> projected,context_cos,context_sin,noise_cos,noise_sin,noise,next,normed,base_logits,corrected;
 std::vector<std::vector<uint16_t>> new_kv,projector_inputs;
 std::vector<uint8_t> mask;
 std::vector<uint16_t> markov_embedding;
 std::string root;
 json spec;
 unsigned position=0;
 // 诊断/消融：关闭 Markov 共现修正时直接用 head 的 base logits 选 token，省掉逐行 markov 执行。
 bool markov_enabled=true;
 std::vector<uint16_t> confidences;

 static size_t elements(const Qnn_Tensor_t&t){size_t n=1;for(unsigned i=0;i<t.v1.rank;i++)n*=t.v1.dimensions[i];return n;}
 static void bind(Qnn_Tensor_t&t,void*data,size_t bytes){t.v1.clientBuf={data,(uint32_t)bytes};}
 void execute(CachedGraph&g,const char*name){check(api.graphExecute(g.graph,g.inputs.data(),g.inputs.size(),g.outputs.data(),g.outputs.size(),nullptr,nullptr),name);}
 void release(){for(auto i=graphs.rbegin();i!=graphs.rend();++i)if(i->context)free_graph(*i);graphs.clear();}
 ~DSparkRuntime(){release();}

 DSparkRuntime(QNN_INTERFACE_VER_TYPE&api_,std::function<CachedGraph(const std::string&,uint32_t,uint32_t)>load,std::function<void(CachedGraph&)>free_,std::string directory,unsigned context_capacity):api(api_),free_graph(free_),capacity(context_capacity),root(std::move(directory)){
  std::ifstream(root+"/config.json")>>spec;hidden=spec.at("hidden_size");dim=spec.at("head_dim");kv_heads=spec.at("num_key_value_heads");layers=spec.at("num_hidden_layers");rows=spec.at("block_size");target_layers=spec.at("target_layer_ids").get<std::vector<unsigned>>();
  if(rows!=7||layers!=5||target_layers.size()!=5)throw std::runtime_error("当前草稿 context 需要 5 层和 block_size=7");
  try{
   for(unsigned i=0;i<4;i++)graphs.push_back(load(root+"/projector_prefix"+std::to_string(i)+".bin",5,1));
   graphs.push_back(load(root+"/projector1.bin",5,1));graphs.push_back(load(root+"/projector8.bin",5,1));
   graphs.push_back(load(root+"/context1.bin",3,2*layers));graphs.push_back(load(root+"/context8.bin",3,2*layers));
   for(unsigned i=0;i<layers;i++)graphs.push_back(load(root+"/layer"+std::to_string(i)+".bin",6,1));
   graphs.push_back(load(root+"/head.bin",1,2));graphs.push_back(load(root+"/markov.bin",4,2));
   markov_embedding=read<uint16_t>(root+"/markov_embedding.u16");if(markov_embedding.size()!=(size_t)spec.at("vocab_size").get<unsigned>()*spec.at("markov_rank").get<unsigned>())throw std::runtime_error("Markov 词表大小错误");keys.assign(layers,std::vector<uint16_t>((size_t)capacity*kv_heads*dim,32768));values=keys;
   projected.resize(8*hidden);context_cos.resize(8*dim);context_sin.resize(8*dim);noise_cos.resize(rows*dim);noise_sin.resize(rows*dim);noise.resize(rows*hidden);next=noise;normed=noise;
   unsigned vocab=spec.at("vocab_size");base_logits.resize(rows*vocab);corrected.resize(vocab);mask.resize(capacity+rows,1);new_kv.assign(2*layers,std::vector<uint16_t>(8*kv_heads*dim));projector_inputs.assign(target_layers.size(),std::vector<uint16_t>(8*hidden));
   for(unsigned i=0;i<layers;i++){require_same_encoding(graphs[6].outputs[2*i],graphs[8+i].inputs[1],"DSpark context K");require_same_encoding(graphs[6].outputs[2*i+1],graphs[8+i].inputs[2],"DSpark context V");if(i+1<layers)require_same_encoding(graphs[8+i].outputs[0],graphs[9+i].inputs[0],"DSpark hidden");}
   require_same_encoding(graphs[8+layers].outputs[0],graphs[9+layers].inputs[3],"DSpark confidence hidden");require_same_encoding(graphs[8+layers].outputs[1],graphs[9+layers].inputs[2],"DSpark Markov logits");
  }catch(...){release();throw;}
 }

 void rope(unsigned first,unsigned count,std::vector<uint16_t>&cosine,std::vector<uint16_t>&sine){
  float theta=spec.at("rope_parameters").at("rope_theta");for(unsigned r=0;r<count;r++)for(unsigned d=0;d<dim/2;d++){float angle=(first+r)/std::pow(theta,2.0f*d/dim);auto c=std::clamp((int)std::round(std::cos(angle)*32767)+32768,0,65535),s=std::clamp((int)std::round(std::sin(angle)*32767)+32768,0,65535);cosine[r*dim+d]=cosine[r*dim+d+dim/2]=c;sine[r*dim+d]=sine[r*dim+d+dim/2]=s;}
 }

 void append(const std::vector<const uint16_t*>&features,unsigned count,unsigned first){
  if(!count||count>8||first!=position||first+count>capacity||features.size()!=target_layers.size())throw std::runtime_error("DSpark KV 追加位置不一致");
  unsigned batch=count==1?1:8;auto&projector=graphs[first<4?first:(batch==1?4:5)];if(first<4&&count!=1)throw std::runtime_error("DSpark 固定前缀必须逐位置建立");auto&context=graphs[batch==1?6:7];
  for(unsigned i=0;i<features.size();i++){std::copy_n(features[i],(size_t)count*hidden,projector_inputs[i].data());std::fill(projector_inputs[i].begin()+(size_t)count*hidden,projector_inputs[i].end(),32768);bind(projector.inputs[i],projector_inputs[i].data(),(size_t)batch*hidden*2);}
  bind(projector.outputs[0],projected.data(),(size_t)batch*hidden*2);execute(projector,"DSpark projector");require_same_encoding(projector.outputs[0],context.inputs[0],"DSpark projector/context");rope(first,batch,context_cos,context_sin);
  bind(context.inputs[0],projected.data(),(size_t)batch*hidden*2);bind(context.inputs[1],context_cos.data(),(size_t)batch*dim*2);bind(context.inputs[2],context_sin.data(),(size_t)batch*dim*2);for(unsigned i=0;i<2*layers;i++)bind(context.outputs[i],new_kv[i].data(),(size_t)batch*kv_heads*dim*2);execute(context,"DSpark context KV");
  for(unsigned l=0;l<layers;l++)for(unsigned r=0;r<count;r++)for(unsigned c=0;c<kv_heads*dim;c++){keys[l][(size_t)c*capacity+first+r]=new_kv[2*l][(size_t)r*kv_heads*dim+c];values[l][((size_t)(c/dim)*capacity+first+r)*dim+c%dim]=new_kv[2*l+1][(size_t)r*kv_heads*dim+c];}
  position+=count;
 }

 void restore_prefix(){
  for(unsigned p=0;p<4;p++){std::vector<std::vector<uint16_t>>owned;std::vector<const uint16_t*>ptrs;for(unsigned layer:target_layers){owned.push_back(read<uint16_t>(root+"/prefix/p"+std::to_string(p)+"_l"+std::to_string(layer)+"_h.u16"));if(owned.back().size()!=hidden)throw std::runtime_error("DSpark 前缀 hidden 尺寸错误");}for(auto&v:owned)ptrs.push_back(v.data());append(ptrs,1,p);}
 }

 // 固定前缀建立后，4 张前缀 projector 图（约 84MB HTP 资源）不再使用，立即释放，
 // 缓解 43 张目标图 + 15 张草稿图共存时的 context 资源 6031/6033。
 void discard_prefix_graphs(){
  for(unsigned i=0;i<4;i++)if(graphs[i].context){free_graph(graphs[i]);graphs[i]=CachedGraph{};}
 }

 void restore(const std::string&directory,unsigned count){
  if(count>capacity)throw std::runtime_error("保存的草稿 KV 超出容量");
  for(unsigned l=0;l<layers;l++){auto k=read<uint16_t>(directory+"/draft_k"+std::to_string(l)+".u16"),v=read<uint16_t>(directory+"/draft_v"+std::to_string(l)+".u16");if(k.size()!=(size_t)count*kv_heads*dim||v.size()!=k.size())throw std::runtime_error("缺少匹配的 DSpark KV，需要重新 prefill 此对话");for(unsigned r=0;r<count;r++)for(unsigned c=0;c<kv_heads*dim;c++){keys[l][(size_t)c*capacity+r]=k[(size_t)r*kv_heads*dim+c];values[l][((size_t)(c/dim)*capacity+r)*dim+c%dim]=v[(size_t)r*kv_heads*dim+c];}}
  position=count;
 }

 void checkpoint(const std::string&directory,const std::function<void(const std::string&,const void*,size_t)>&write){
  std::vector<uint16_t>k((size_t)position*kv_heads*dim),v(k.size());for(unsigned l=0;l<layers;l++){for(unsigned r=0;r<position;r++)for(unsigned c=0;c<kv_heads*dim;c++){k[(size_t)r*kv_heads*dim+c]=keys[l][(size_t)c*capacity+r];v[(size_t)r*kv_heads*dim+c]=values[l][((size_t)(c/dim)*capacity+r)*dim+c%dim];}write(directory+"/draft_k"+std::to_string(l)+".u16",k.data(),k.size()*2);write(directory+"/draft_v"+std::to_string(l)+".u16",v.data(),v.size()*2);}
 }

 std::vector<uint32_t> propose(const uint16_t*embedding,uint32_t bonus,unsigned max_proposals=7){
  std::copy_n(embedding,(size_t)rows*hidden,noise.begin());rope(position,rows,noise_cos,noise_sin);std::fill(mask.begin(),mask.end(),1);std::fill_n(mask.begin(),position,0);std::fill(mask.begin()+capacity,mask.end(),0);
  for(unsigned l=0;l<layers;l++){auto&g=graphs[8+l];bind(g.inputs[0],noise.data(),noise.size()*2);bind(g.inputs[1],keys[l].data(),keys[l].size()*2);bind(g.inputs[2],values[l].data(),values[l].size()*2);bind(g.inputs[3],noise_cos.data(),noise_cos.size()*2);bind(g.inputs[4],noise_sin.data(),noise_sin.size()*2);bind(g.inputs[5],mask.data(),mask.size());bind(g.outputs[0],next.data(),next.size()*2);execute(g,"DSpark draft layer");noise.swap(next);}
  auto&head=graphs[8+layers];bind(head.inputs[0],noise.data(),noise.size()*2);bind(head.outputs[0],normed.data(),normed.size()*2);bind(head.outputs[1],base_logits.data(),base_logits.size()*2);execute(head,"DSpark draft head");
  std::vector<uint32_t>proposal;unsigned vocab=spec.at("vocab_size");confidences.assign(rows,0);
  if(!markov_enabled){
   // 消融路径：逐行取 base logits 的 argmax，前一行结果只用于下一行的链式语义记录。
   for(unsigned r=0;r<std::min(rows,max_proposals);r++){auto first=base_logits.begin()+(size_t)r*vocab;bonus=std::max_element(first,first+vocab)-first;proposal.push_back(bonus);}
   return proposal;
  }
  auto&markov=graphs[9+layers];uint16_t confidence=0;
  for(unsigned r=0;r<std::min(rows,max_proposals);r++){uint32_t group=bonus/32,index=bonus%32;unsigned rank=spec.at("markov_rank");bind(markov.inputs[0],markov_embedding.data()+(size_t)group*32*rank,(size_t)32*rank*2);bind(markov.inputs[1],&index,4);bind(markov.inputs[2],base_logits.data()+(size_t)r*vocab,(size_t)vocab*2);bind(markov.inputs[3],normed.data()+(size_t)r*hidden,(size_t)hidden*2);bind(markov.outputs[0],corrected.data(),corrected.size()*2);bind(markov.outputs[1],&confidence,2);execute(markov,("DSpark Markov head draft_row="+std::to_string(r)+" previous="+std::to_string(bonus)+" group="+std::to_string(group)+" row="+std::to_string(index)).c_str());bonus=std::max_element(corrected.begin(),corrected.end())-corrected.begin();proposal.push_back(bonus);confidences[r]=confidence;}
  return proposal;
 }
};
