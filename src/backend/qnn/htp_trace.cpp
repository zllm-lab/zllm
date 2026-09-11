// trace 驱动的全算子 HTP 图构建与执行 runner(原 zllm-bench qnn_htp_replay)。
// 模型结构与量化配置完全来自 root 下的 trace.jsonl 与权重文件,本文件不含任何
// 模型或 HTP 架构分支;MiniCPM5 1B 验收与 SenseVoice 融合段共用此入口。
// 由 zllm-qnn-trace bin 经 C ABI 调用,保持原有 stdout 协议(TOKEN/RESIDENT_*/HOT)。
#include "htp_graph.hpp"
#include "htp_session.hpp"
#include <sys/stat.h>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>

extern "C" int zllm_qnn_trace_main(int argc,char**argv){try{
 if(argc!=3||std::string(argv[1])!="--config")return 2;std::ifstream cf(argv[2]);json cfg;cf>>cfg;std::string root=cfg.at("root");
 auto session=htp_session_open(cfg);
 auto&api=session.api;Qnn_BackendHandle_t backend=session.backend;Qnn_DeviceHandle_t device=session.device;
 std::ifstream tf(root+"/trace.jsonl");std::vector<json>trace;std::map<int,json>meta;std::string line;while(std::getline(tf,line)){auto j=json::parse(line);trace.push_back(j);if(j.contains("id"))meta[j["id"].get<int>()]=j;}
 if(cfg.value("resident_layer",false)){
  unsigned model_layers=cfg.value("model_layers",24u);int decode_stride=15*model_layers+3,hidden_id=decode_stride+1,layer_base=hidden_id+1,final_norm_id=layer_base+15*model_layers;
  bool continuous=cfg.value("resident_continuous",false);
  auto paired=[&](int id,const char*field){float value=meta.at(id)[field].get<float>();if(continuous&&meta.count(id+decode_stride))value=std::max(value,meta.at(id+decode_stride)[field].get<float>());return value;};
  auto step_for=[&](int id){return std::max(paired(id,"max")*cfg.value("range_margin",1.1f)/32767,1e-8f);};
  auto value_for=[&](int id){auto&m=meta.at(id);auto reference=read<float>(root+"/t"+std::to_string(id)+".f32");Value value{std::vector<uint16_t>(reference.size()),m["rows"],m["cols"],step_for(id)};for(size_t i=0;i<reference.size();i++)value.data[i]=std::clamp((int)std::round(reference[i]/value.step)+32768,0,65535);return value;};
  unsigned resident_layers=cfg.value("resident_layers",1u);if(resident_layers<1||resident_layers>model_layers)throw std::runtime_error("resident_layers 超出模型层数");bool resident_output=cfg.value("resident_output",false);if(resident_output&&resident_layers!=model_layers)throw std::runtime_error("resident_output 要求执行全部模型层");if(continuous&&!resident_output)throw std::runtime_error("连续解码要求 resident_output");
  uint32_t capacity=continuous?cfg.value("resident_kv_capacity",32u):19u;if(capacity<20)throw std::runtime_error("resident_kv_capacity 至少为 20");int final_id=layer_base+15*(resident_layers-1)+14;Value hidden=value_for(hidden_id);uint32_t hidden_size=hidden.cols;Value result{std::vector<uint16_t>(hidden_size),1,hidden_size,step_for(final_id)};auto&attention_spec=meta.at(layer_base+6);uint32_t attention_heads=attention_spec["heads"],kv_heads=attention_spec["kv_heads"],head_dim=attention_spec["dim"],kv_cols=kv_heads*head_dim;if(attention_heads%kv_heads)throw std::runtime_error("attention heads 不能整除 KV heads");
  Graph graph(api,backend,device);auto hidden_tensor=graph.input(hidden);
  auto native=[&](int id){auto&m=meta.at(id);return graph.tensor({m["rows"],m["cols"]},step_for(id));};
  auto norm=[&](Qnn_Tensor_t input,int id){auto&j=meta.at(id);uint32_t n=j["cols"],d=n;auto x=graph.reshape(input,{1,d});float xs=input.v1.quantizeParams.scaleOffsetEncoding.scale;auto epsilon=graph.constant({std::sqrt(d*j["eps"].get<float>())},{1,1},xs);auto eps_same=graph.tensor({1,1},xs);graph.op(QNN_OP_CONVERT,{epsilon},eps_same);auto padded=graph.tensor({1,d+1},xs);graph.op(QNN_OP_CONCAT,{x,eps_same},padded,{graph.scalar("axis",1)});auto unit=graph.tensor({1,d+1},paired(id,"normalized_step")/std::sqrt((float)d));auto eps=graph.scalar("epsilon",0,QNN_DATATYPE_FLOAT_32);eps.scalarParam.floatValue=1e-12f;graph.op(QNN_OP_L2_NORM,{padded},unit,{graph.scalar("axis",1),eps});auto normalized=graph.slice(unit,0,d);auto weights=read<float>(root+"/w"+std::to_string(j["weight"].get<int>())+".f32");for(auto&v:weights)v=(v+j["offset"].get<float>())*std::sqrt((float)d);auto gamma=graph.constant(weights,{1,(uint32_t)weights.size()});auto output=native(id);graph.op(QNN_OP_ELEMENT_WISE_MULTIPLY,{normalized,gamma},output);return output;};
  auto linear=[&](Qnn_Tensor_t input,int id){auto&j=meta.at(id);uint32_t k=input.v1.dimensions[1],n=j["cols"];int wid=j["weight"],bits=cfg.value("resident_weight_bits",4);if(cfg.contains("resident_eight_bit_weights"))for(int candidate:cfg["resident_eight_bit_weights"])if(candidate==wid)bits=8;std::string stem=bits==8?".pc8":".pc4awq";auto output=native(id);auto weights=read<uint8_t>(root+"/w"+std::to_string(wid)+stem);auto scales=read<float>(root+"/w"+std::to_string(wid)+stem+"scales");if(weights.size()!=(size_t)k*n||scales.size()!=n)throw std::runtime_error("常驻低比特权重尺寸不匹配 w"+std::to_string(wid));graph.buffers.push_back(std::move(weights));auto weight=graph.lowbit({k,n},graph.buffers.back().data(),graph.buffers.back().size(),std::move(scales),bits);graph.op(QNN_OP_MAT_MUL,{input,weight},output);return output;};

  Value rope_cos{std::vector<uint16_t>(head_dim),1,head_dim,1.0f/32767},rope_sin=rope_cos;
  auto fill_rope=[&](unsigned position){std::string table="/rope_"+std::to_string(head_dim)+"_19_1_";auto cosine=read<float>(root+table+"cos.f32"),sine=read<float>(root+table+"sin.f32");for(unsigned d=0;d<head_dim/2;d++){auto cq=std::clamp((int)std::round(cosine[position*(head_dim/2)+d]/rope_cos.step)+32768,0,65535),sq=std::clamp((int)std::round(sine[position*(head_dim/2)+d]/rope_sin.step)+32768,0,65535);rope_cos.data[2*d]=rope_cos.data[2*d+1]=cq;rope_sin.data[2*d]=rope_sin.data[2*d+1]=sq;}};
  fill_rope(19);auto cosine_tensor=graph.input(rope_cos),sine_tensor=graph.input(rope_sin);
  auto rope=[&](Qnn_Tensor_t input,int id){auto&j=meta.at(id);uint32_t n=j["cols"],heads=j["heads"],hd=n/heads;auto x=graph.reshape(input,{heads,hd});std::vector<float> permutation((size_t)hd*hd,0);for(unsigned d=0;d<hd/2;d++){unsigned a=2*d,b=2*d+1;permutation[(size_t)b*hd+a]=-1;permutation[(size_t)a*hd+b]=1;}float xs=input.v1.quantizeParams.scaleOffsetEncoding.scale;auto perm=graph.constant(permutation,{hd,hd});auto rotated=graph.tensor({heads,hd},xs);graph.op(QNN_OP_MAT_MUL,{x,perm},rotated);auto real=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,x,cosine_tensor,{heads,hd},xs),imaginary=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,rotated,sine_tensor,{heads,hd},xs);auto sum=graph.binary(QNN_OP_ELEMENT_WISE_ADD,real,imaginary,{heads,hd},step_for(id));auto output=native(id);auto flat=graph.reshape(sum,{1,n});graph.op(QNN_OP_CONVERT,{flat},output);return output;};

  std::vector<uint8_t>attention_mask(capacity+1,0);auto set_mask=[&](unsigned position){std::fill(attention_mask.begin(),attention_mask.end(),0);for(unsigned row=position;row<capacity;row++)attention_mask[row]=1;};set_mask(19);auto mask_tensor=graph.tensor({1,capacity+1},0,QNN_TENSOR_TYPE_APP_WRITE,nullptr,0,QNN_DATATYPE_BOOL_8);mask_tensor.v1.clientBuf={attention_mask.data(),(uint32_t)attention_mask.size()};graph.ins.push_back(mask_tensor);
  std::deque<Value>kv_inputs,kv_outputs;Qnn_Tensor_t current=hidden_tensor;
  for(unsigned layer=0;layer<resident_layers;layer++){
   int b=layer_base+15*layer,pk=6+15*layer,pv=4+15*layer;
   auto prefix_key=read<float>(root+"/t"+std::to_string(pk)+".f32"),prefix_value=read<float>(root+"/t"+std::to_string(pv)+".f32");
   float key_step=std::max(std::max(meta.at(pk)["max"].get<float>(),paired(b+5,"max"))*cfg.value("range_margin",1.1f)/32767,1e-8f),value_step=std::max(std::max(meta.at(pv)["max"].get<float>(),paired(b+3,"max"))*cfg.value("range_margin",1.1f)/32767,1e-8f);
   kv_inputs.push_back(Value{std::vector<uint16_t>((size_t)capacity*kv_cols,32768),capacity,kv_cols,key_step});for(size_t i=0;i<prefix_key.size();i++)kv_inputs.back().data[i]=std::clamp((int)std::round(prefix_key[i]/key_step)+32768,0,65535);auto old_key=graph.input(kv_inputs.back());
   kv_inputs.push_back(Value{std::vector<uint16_t>((size_t)capacity*kv_cols,32768),capacity,kv_cols,value_step});for(size_t i=0;i<prefix_value.size();i++)kv_inputs.back().data[i]=std::clamp((int)std::round(prefix_value[i]/value_step)+32768,0,65535);auto old_value=graph.input(kv_inputs.back());
   auto norm1=norm(current,b),query=rope(linear(norm1,b+1),b+4),new_key=rope(linear(norm1,b+2),b+5),new_value=linear(norm1,b+3);
   auto new_key_same=graph.tensor({1,kv_cols},key_step),new_value_same=graph.tensor({1,kv_cols},value_step);graph.op(QNN_OP_CONVERT,{new_key},new_key_same);graph.op(QNN_OP_CONVERT,{new_value},new_value_same);
   kv_outputs.push_back(Value{std::vector<uint16_t>(kv_cols),1,kv_cols,key_step});graph.op(QNN_OP_CONVERT,{new_key_same},graph.output(kv_outputs.back()));kv_outputs.push_back(Value{std::vector<uint16_t>(kv_cols),1,kv_cols,value_step});graph.op(QNN_OP_CONVERT,{new_value_same},graph.output(kv_outputs.back()));
   auto keys=graph.tensor({capacity+1,kv_cols},key_step),values=graph.tensor({capacity+1,kv_cols},value_step);graph.op(QNN_OP_CONCAT,{old_key,new_key_same},keys,{graph.scalar("axis",0)});graph.op(QNN_OP_CONCAT,{old_value,new_value_same},values,{graph.scalar("axis",0)});
   uint32_t queries_per_kv=attention_heads/kv_heads;std::vector<Qnn_Tensor_t>heads;for(unsigned h=0;h<attention_heads;h++){auto q=graph.slice(query,h*head_dim,head_dim),k=graph.slice(keys,(h/queries_per_kv)*head_dim,head_dim),v=graph.slice(values,(h/queries_per_kv)*head_dim,head_dim);auto kt=graph.tensor({head_dim,capacity+1},key_step);graph.op(QNN_OP_TRANSPOSE,{k},kt,{graph.array("perm",{1,0})});float score_step=(2.0f*head_dim)/32767;auto scores=graph.tensor({1,capacity+1},score_step);graph.op(QNN_OP_MAT_MUL,{q,kt},scores);auto masked=graph.tensor({1,capacity+1},score_step);graph.op(QNN_OP_ELEMENT_WISE_SELECT,{mask_tensor,graph.constant(-32768*score_step),scores},masked);auto probability=graph.tensor({1,capacity+1},1.0f/65535,QNN_TENSOR_TYPE_NATIVE,nullptr,0,QNN_DATATYPE_UFIXED_POINT_16,0);auto beta=graph.scalar("beta",0,QNN_DATATYPE_FLOAT_32);beta.scalarParam.floatValue=meta.at(b+6)["score_scale"].get<float>();graph.op(QNN_OP_SOFTMAX,{masked},probability,{graph.scalar("axis",1),beta});auto head=graph.tensor({1,head_dim},step_for(b+6));graph.op(QNN_OP_MAT_MUL,{probability,v},head);heads.push_back(head);}
   auto attention=native(b+6);graph.op(QNN_OP_CONCAT,heads,attention,{graph.scalar("axis",1)});auto projected=linear(attention,b+7),residual1=native(b+8);graph.op(QNN_OP_ELEMENT_WISE_ADD,{current,projected},residual1);auto norm2=norm(residual1,b+9),gate=linear(norm2,b+10),up=linear(norm2,b+11);uint32_t intermediate=meta.at(b+10)["cols"];auto sigmoid=graph.tensor({1,intermediate},1.0f/65535,QNN_TENSOR_TYPE_NATIVE,nullptr,0,QNN_DATATYPE_UFIXED_POINT_16,0);graph.op(QNN_OP_SIGMOID,{gate},sigmoid);auto silu=graph.tensor({1,intermediate},step_for(b+10));graph.op(QNN_OP_ELEMENT_WISE_MULTIPLY,{gate,sigmoid},silu);auto activation=native(b+12);graph.op(QNN_OP_ELEMENT_WISE_MULTIPLY,{silu,up},activation);auto down=linear(activation,b+13);auto output=layer+1==resident_layers&&!resident_output?graph.output(result):native(b+14);graph.op(QNN_OP_ELEMENT_WISE_ADD,{residual1,down},output);current=output;
  }
  uint32_t resident_token=0;Value resident_logits{};if(resident_output){auto final_norm=norm(current,final_norm_id),logits=linear(final_norm,final_norm_id+1);auto&m=meta.at(final_norm_id+1);resident_logits={std::vector<uint16_t>((size_t)m["rows"].get<unsigned>()*m["cols"].get<unsigned>()),m["rows"],m["cols"],step_for(final_norm_id+1)};graph.op(QNN_OP_CONVERT,{logits},graph.output(resident_logits));}
  auto expected=cfg.value("resident_expected_tokens",std::vector<unsigned>{15963,13593});auto begin=std::chrono::steady_clock::now();graph.execute(cfg.value("repeats",100u));double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();if(resident_output){resident_token=argmax(resident_logits);if(expected.empty())throw std::runtime_error("resident_expected_tokens 不能为空");printf("RESIDENT_DECODE position=19 token=%u expected=%u wall_ms=%.3f\n",resident_token,expected[0],wall);}else{auto reference=read<float>(root+"/t"+std::to_string(final_id)+".f32");float max_error=0;for(size_t i=0;i<reference.size();i++)max_error=std::max(max_error,std::abs(((int)result.data[i]-32768)*result.step-reference[i]));printf("RESIDENT_LAYERS count=%u wall_ms=%.3f maxerr=%.6f\n",resident_layers,wall,max_error);}
  bool failed=resident_output&&resident_token!=expected[0];
  if(continuous&&!failed){
   unsigned decode_steps=cfg.value("resident_decode_steps",2u);if(decode_steps<2||19+decode_steps>capacity||expected.size()<decode_steps)throw std::runtime_error("resident_decode_steps 超出 KV capacity 或 expected token 数量");
   for(unsigned position=20;position<19+decode_steps;position++){
    for(unsigned layer=0;layer<resident_layers;layer++){auto&key=kv_inputs[2*layer];auto&value=kv_inputs[2*layer+1];std::copy(kv_outputs[2*layer].data.begin(),kv_outputs[2*layer].data.end(),key.data.begin()+(position-1)*kv_cols);std::copy(kv_outputs[2*layer+1].data.begin(),kv_outputs[2*layer+1].data.end(),value.data.begin()+(position-1)*kv_cols);}
    Value next_hidden{std::vector<uint16_t>(hidden_size),1,hidden_size,hidden.step};Graph embedding_graph(api,backend,device);auto embedding_output=embedding_graph.output(next_hidden);std::ifstream mf(root+"/embedding.json");json spec;mf>>spec;uint32_t block=spec["block_rows"],vocab=spec["rows"];if(resident_token>=vocab)throw std::runtime_error("连续解码 token 越界");unsigned group=resident_token/block;std::ifstream data(root+"/embedding.u16",std::ios::binary),step_file(root+"/embedding.steps",std::ios::binary);float scale;step_file.seekg(group*4);step_file.read((char*)&scale,4);Value table{std::vector<uint16_t>((size_t)block*hidden_size),block,hidden_size,scale};data.seekg((uint64_t)group*block*hidden_size*2);data.read((char*)table.data.data(),table.data.size()*2);if(!data||!step_file)throw std::runtime_error("连续解码 embedding 权重块读取失败");auto selected=embedding_graph.slice(embedding_graph.input(table),resident_token%block,1,0);embedding_graph.op(QNN_OP_CONVERT,{selected},embedding_output);embedding_graph.execute();std::copy(next_hidden.data.begin(),next_hidden.data.end(),hidden.data.begin());fill_rope(position);set_mask(position);graph.execute();resident_token=argmax(resident_logits);unsigned wanted=expected[position-19];printf("RESIDENT_CONTINUOUS position=%u token=%u expected=%u\n",position,resident_token,wanted);failed=failed||resident_token!=wanted;
   }
  }
  return failed;
 }
 std::map<int,std::unique_ptr<Value>>values;std::map<int,std::pair<std::unique_ptr<Value>,std::unique_ptr<Value>>>cache;
 float audit_atol=cfg.value("audit_atol",0.01f),audit_rtol=cfg.value("audit_rtol",0.01f);
 int start=cfg.value("start_at",0),stop=cfg.value("stop_after",2147483647);unsigned stages=0,failures=0;uint32_t last_token=0;bool generated=false;
 std::vector<unsigned> prompt_tokens=cfg.value("prompt_tokens",std::vector<unsigned>{});
 std::vector<unsigned> graph_tokens=prompt_tokens;unsigned prompt_rows=cfg.value("prompt_capacity",(unsigned)prompt_tokens.size());if(!prompt_tokens.empty()){if(prompt_rows<prompt_tokens.size())throw std::runtime_error("prompt_capacity 小于真实 prompt");graph_tokens.resize(prompt_rows,cfg.value("pad_token",0u));}
 bool chat=!prompt_tokens.empty(),audit=cfg.value("audit",!chat),require_expected=cfg.value("require_expected",!chat);
 std::string audit_root=cfg.value("audit_root",root);
 unsigned calibration_rows=0;for(const auto&j:trace)if(j.value("op",std::string())=="input"&&j.value("kind",std::string())=="embedding"){calibration_rows=j.value("rows",0u);break;}if(chat&&!calibration_rows)throw std::runtime_error("trace 缺少 prompt 行数");
 int prompt_delta=chat?(int)prompt_tokens.size()-(int)calibration_rows:0;unsigned generation_index=0;
 // SenseVoice 融合模式：trace 按 segment_ops 个算子分段编译成多张图，段间以
 // U16 buffer 交接（APP 往返从每 op 一次降为每段一次）；末段 logits 读回后
 // host 完成 argmax + CTC 贪心 + tokens 拼接。逐段审计，硬门禁是逐帧 id 一致。
 bool fused=cfg.value("sensevoice_resident",false);
 int segment_ops=fused?cfg.value("segment_ops",25):0;
 auto fused_begin=std::chrono::steady_clock::now();
 std::unique_ptr<Graph> fused_graph_holder;
 Graph *fused_graph_ptr=nullptr;
 std::map<int,Qnn_Tensor_t>fused_tensors;
 std::vector<std::tuple<int,std::unique_ptr<Value>,std::unique_ptr<Value>>> fused_cache;
 std::vector<std::tuple<int,float,std::unique_ptr<Value>>> fused_outputs;int seg_last_id=-1;
 int final_id=-1;
 std::set<int> cut_before;
 std::map<int,int> max_consumer,segment_for;
 if(fused){
  for(auto&t:trace)if(t.contains("id")&&t["op"]!="argmax")final_id=t["id"];
  if(final_id<0)throw std::runtime_error("trace 为空");
  // 段切分约束：段内除末 op 外的所有张量的消费者必须落在段内（否则跨段引用
  // 已销毁图中的 tensor）。先用 max_consumer 预分析，再贪心在单边交接处切。
  std::vector<json*> ops;for(auto&t:trace)if(t.contains("id")&&t["op"]!="argmax")ops.push_back(&t);
  for(auto*t:ops){int tid=(*t)["id"];for(int ii:(*t)["inputs"])max_consumer[ii]=std::max(max_consumer.count(ii)?max_consumer[ii]:ii,tid);}
  size_t start=0;
  while(start<ops.size()){
   size_t j=start;
   while(j+1<ops.size()){
    int candidate=(*ops[j])["id"];
    if(j-start+1>=segment_ops){
     json &next=*ops[j+1];int nid=next["id"];
     bool single_edge=true;
     for(int ii:next["inputs"])single_edge=single_edge&&ii==candidate;
     if(single_edge){
      bool internal=true;
      for(size_t k=start;k<j;k++){int tk=(*ops[k])["id"];if(max_consumer[tk]>candidate){internal=false;break;}}
      if(internal)break;
     }
    }
    j++;
   }
   if(j+1<ops.size())cut_before.insert((*ops[j+1])["id"].get<int>());
   start=j+1;
  }
  if(cfg.value("split_attention",false))for(auto*t:ops)if((*t)["op"]=="attention"){int id=(*t)["id"];cut_before.insert(id);cut_before.insert(id+1);}
  int segment=0;for(auto*t:ops){int id=(*t)["id"];if(cut_before.count(id))segment++;segment_for[id]=segment;}
  printf("SEGMENTS cuts=%zu\n",cut_before.size());fflush(stdout);
 }
 auto close_segment=[&](){
  if(!fused_graph_ptr)return;
  fused_graph_ptr->execute(cfg.value("repeats",1u));
  if(cfg.contains("context_cache_dir")){std::string dir=cfg["context_cache_dir"];mkdir(dir.c_str(),0755);fused_graph_ptr->save(dir+"/segment_"+std::to_string(seg_last_id)+".bin");}
  float maxerr=0;unsigned bad=0;
  for(auto& item:fused_outputs){int id=std::get<0>(item);float item_step=std::get<1>(item);auto& value=*std::get<2>(item);if(audit){auto reference=read<float>(audit_root+"/t"+std::to_string(id)+".f32");if(reference.size()!=value.data.size())throw std::runtime_error("融合段审计尺寸不匹配");for(size_t i=0;i<value.data.size();i++){float actual=((int)value.data[i]-32768)*item_step;float e=std::abs(actual-reference[i]);maxerr=std::max(maxerr,e);bad+=e>audit_atol+audit_rtol*std::abs(reference[i]);}}values[id]=std::move(std::get<2>(item));}fused_outputs.clear();
  printf("segment=%d maxerr=%.6f bad=%u\n",seg_last_id,maxerr,bad);fflush(stdout);
  for(auto& pending:fused_cache){int layer=std::get<0>(pending);cache[layer]={std::move(std::get<1>(pending)),std::move(std::get<2>(pending))};}fused_cache.clear();
  failures+=bad!=0;stages++;
  fused_graph_holder.reset();fused_graph_ptr=nullptr;fused_tensors.clear();
 };

 for(auto &j:trace){std::string op=j["op"];
  // 聊天模式复用离线校准的算子/权重元数据，只把序列维和位置改为本次真实 prompt。
  // 量化范围仍来自多步 calibration；CPU 不参与任何模型数值算子。
  if(chat){
   if(generation_index==0){
    if(j.contains("rows")&&j["rows"].get<unsigned>()==calibration_rows)j["rows"]=prompt_rows;
    if(op=="input"&&j.value("kind",std::string())=="embedding")j["tokens"]=graph_tokens;
    if(op=="row")j["row"]=(unsigned)prompt_tokens.size()-1;
   }else if((op=="rope"||op=="attention"||op=="attention_shared")&&j.contains("position"))j["position"]=j["position"].get<int>()+prompt_delta;
  }
  if(op=="argmax"){
  int input_id=j["inputs"][0];if(!values.count(input_id))continue;auto &logits=*values.at(input_id);last_token=argmax(logits,j["excluded"].get<std::vector<unsigned>>());generated=true;
  unsigned expected=j.value("expected_token",last_token);printf("TOKEN actual=%u expected=%u\n",last_token,expected);fflush(stdout);
  if(generation_index==0&&cfg.contains("state_dir")){
   std::string dir=cfg["state_dir"];mkdir(dir.c_str(),0755);json state={{"prompt_tokens",prompt_tokens.size()},{"first_token",last_token},{"layers",cache.size()}};
   for(auto&[layer,kv]:cache){write(dir+"/k"+std::to_string(layer)+".u16",kv.first->data);write(dir+"/v"+std::to_string(layer)+".u16",kv.second->data);state["kv"][std::to_string(layer)]={{"rows",kv.first->rows},{"cols",kv.first->cols},{"key_step",kv.first->step},{"value_step",kv.second->step}};}
   std::ofstream sf(dir+"/state.json");sf<<state.dump();if(!sf)throw std::runtime_error("write "+dir+"/state.json");
  }
  generation_index++;if(require_expected&&last_token!=expected)throw std::runtime_error("端到端 token 不一致");continue;
 }int id=j["id"];if(id<start){if(cfg.value("seed_skipped",false)){uint32_t sr=j["rows"],sn=j["cols"];float ss=std::max(j["max"].get<float>()*cfg.value("range_margin",1.1f)/32767,1e-8f);auto reference=read<float>(root+"/t"+std::to_string(id)+".f32");auto seeded=std::make_unique<Value>(Value{std::vector<uint16_t>(sr*sn),sr,sn,ss});for(size_t i=0;i<reference.size();i++)seeded->data[i]=std::clamp((int)std::round(reference[i]/ss)+32768,0,65535);values[id]=std::move(seeded);}continue;}if(id>stop)break;uint32_t r=j["rows"],n=j["cols"];float range=j["max"].get<float>();if(cfg.contains("range_cap"))range=std::min(range,cfg["range_cap"].get<float>());float margin=cfg.value("range_margin",1.1f);float step=std::max(range*margin/32767,1e-8f);auto out=std::make_unique<Value>(Value{std::vector<uint16_t>(r*n),r,n,step});if(id>=cfg.value("rowwise_from",2147483647)&&id<=cfg.value("rowwise_through",-1)){auto reference=read<float>(audit_root+"/t"+std::to_string(id)+".f32");out->row_steps.resize(r,1e-8f);for(unsigned row=0;row<r;row++){float peak=0;for(unsigned col=0;col<n;col++)peak=std::max(peak,std::abs(reference[(size_t)row*n+col]));out->row_steps[row]=std::max(peak*margin/32767,1e-8f);}}
  if(op=="input"){
   Graph graph(api,backend,device);auto result=graph.output(*out);std::string kind=j["kind"];std::ifstream mf(root+"/"+kind+".json");
   if(!mf){
    // 非 embedding 输入（SenseVoice 组装好的 fbank/PE 行）：审计模式用参考 t{id}.f32
    // 量化播种；该文件就是 host 前端的真实输出，不是设备侧数值来源。
    auto reference=read<float>(root+"/t"+std::to_string(id)+".f32");if(reference.size()!=out->data.size())throw std::runtime_error("输入参考尺寸不匹配");
    for(size_t i=0;i<reference.size();i++)out->data[i]=std::clamp((int)std::round(reference[i]/step)+32768,0,65535);
    values[id]=std::move(out);printf("stage=%d op=input rows=%u cols=%u\n",id,r,n);fflush(stdout);continue;
   }
   json spec;mf>>spec;uint32_t block=spec["block_rows"],vocab=spec["rows"];std::ifstream data(root+"/"+kind+".u16",std::ios::binary),step_file(root+"/"+kind+".steps",std::ios::binary);std::deque<Value> tables;std::vector<Qnn_Tensor_t> rows;
   for(unsigned row=0;row<r;row++){unsigned token=r==1&&generated?last_token:j["tokens"][row].get<unsigned>();if(token>=vocab)throw std::runtime_error("token 越界");unsigned group=token/block;float scale;step_file.seekg(group*4);step_file.read((char*)&scale,4);tables.push_back(Value{std::vector<uint16_t>(block*n),block,n,scale});if(cfg.value("embedding_block_files",false)){tables.back().data=read<uint16_t>(root+"/"+kind+"_"+std::to_string(group)+".u16");}else{data.seekg((uint64_t)group*block*n*2);data.read((char*)tables.back().data.data(),block*n*2);}if((!cfg.value("embedding_block_files",false)&&!data)||!step_file)throw std::runtime_error("embedding 权重块读取失败");auto weight=graph.input(tables.back());auto selected=graph.slice(weight,token%block,1,0);auto converted=graph.tensor({1,n},step);graph.op(QNN_OP_CONVERT,{selected},converted);rows.push_back(converted);}
   if(rows.size()==1)graph.op(QNN_OP_CONVERT,{rows[0]},result);else graph.op(QNN_OP_CONCAT,rows,result,{graph.scalar("axis",0)});graph.execute();if(cfg.contains("dump_dir"))write(cfg["dump_dir"].get<std::string>()+"/t"+std::to_string(id)+".u16",out->data);float error=0;if(audit){auto reference=read<float>(audit_root+"/t"+std::to_string(id)+".f32");for(unsigned z=0;z<out->data.size();z++)error=std::max(error,std::abs(((int)out->data[z]-32768)*step-reference[z]));}values[id]=std::move(out);printf("stage=%d op=embedding_gather rows=%u maxerr=%.6f\n",id,r,error);fflush(stdout);continue;
  }
  auto start=std::chrono::steady_clock::now();
  if(cfg.value("trace_ops",false)){printf("OP id=%d op=%s\n",id,op.c_str());fflush(stdout);}
  bool boundary=!fused||cut_before.count(id);
  bool seg_end=fused&&(id==final_id||cut_before.count(id+1));
  if(fused&&boundary)close_segment();
  std::unique_ptr<Graph> stage_graph;
  if(!fused)stage_graph=std::make_unique<Graph>(api,backend,device);
  else if(!fused_graph_ptr){fused_graph_holder=std::make_unique<Graph>(api,backend,device);fused_graph_ptr=fused_graph_holder.get();}
  Graph &graph=fused?*fused_graph_ptr:*stage_graph;
  std::vector<Qnn_Tensor_t> in;
  for(int ii:j["inputs"])in.push_back(fused&&fused_tensors.count(ii)?fused_tensors.at(ii):graph.input(*values.at(ii)));
  std::vector<uint32_t>shape={r,n};
  Value* output_value=out.get();
  Qnn_Tensor_t output;
  bool export_output=fused&&(seg_end||(max_consumer.count(id)&&segment_for[max_consumer.at(id)]>segment_for[id]));
  if(export_output){output=graph.output(*out);fused_outputs.emplace_back(id,step,std::move(out));seg_last_id=id;}
  else if(fused)output=graph.tensor(shape,step);
  else output=graph.output(*out);
  std::unique_ptr<Value> next_k,next_v;int next_layer=-1;
  if(op=="slice"||op=="row"){auto selected=graph.slice(in[0],j.value(op=="slice"?"start":"row",0),op=="slice"?n:r,op=="slice"?1:0);graph.op(QNN_OP_CONVERT,{selected},output);}
  else if(op=="add"){float scale=j["scale"].get<float>();auto s=output_value->row_steps.empty()?graph.binary(QNN_OP_ELEMENT_WISE_ADD,in[0],in[1],shape,step/std::max(std::abs(scale),1e-8f)):graph.binary_rows(QNN_OP_ELEMENT_WISE_ADD,in[0],in[1],shape,output_value->row_steps);graph.op(QNN_OP_ELEMENT_WISE_MULTIPLY,{s,graph.constant(scale)},output);}
  else if(op=="norm"){
   uint32_t h=j["heads"],d=n/h;auto x=graph.reshape(in[0],{r*h,d});float xs=in[0].v1.quantizeParams.scaleOffsetEncoding.scale;
   // 追加 sqrt(d*eps)，使 L2 分母精确对应 RMSNorm 的 sum(x²)+d*eps。
   auto epsilon=graph.constant(std::vector<float>(r*h,std::sqrt(d*j["eps"].get<float>())),{r*h,1},xs);
   // Concat 前显式统一 offset 和 scale，避免不同量化编码被直接拼接。
   auto eps_same=graph.tensor({r*h,1},xs);graph.op(QNN_OP_CONVERT,{epsilon},eps_same);auto padded=graph.tensor({r*h,d+1},xs);graph.op(QNN_OP_CONCAT,{x,eps_same},padded,{graph.scalar("axis",1)});
   auto unit=graph.tensor({r*h,d+1},j["normalized_step"].get<float>()/std::sqrt((float)d));auto eps=graph.scalar("epsilon",0,QNN_DATATYPE_FLOAT_32);eps.scalarParam.floatValue=1e-12f;graph.op(QNN_OP_L2_NORM,{padded},unit,{graph.scalar("axis",1),eps});auto normalized=graph.slice(unit,0,d);
   auto weights=read<float>(root+"/w"+std::to_string(j["weight"].get<int>())+".f32");for(auto&v:weights)v=(v+j["offset"].get<float>())*std::sqrt((float)d);auto gamma=graph.constant(weights,{1,(uint32_t)weights.size()});auto y=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,normalized,gamma,{r*h,d},step);auto flat=graph.reshape(y,shape);graph.op(QNN_OP_CONVERT,{flat},output);

  }
  else if(op=="linear"){
   bool exact_group=cfg.value("weight_format",std::string())=="groupwise_i8";if(cfg.contains("groupwise_i8_weights"))for(int candidate:cfg["groupwise_i8_weights"])exact_group=exact_group||candidate==j["weight"].get<int>();
   if(exact_group){
    int wid=j["weight"];uint32_t k=in[0].v1.dimensions[1],group=k%64==0?64:k;auto weights=read<uint8_t>(root+"/w"+std::to_string(wid)+".i8");auto scales=read<float>(root+"/w"+std::to_string(wid)+".scales");if(weights.size()!=(size_t)k*n||scales.size()!=(size_t)(k/group)*n)throw std::runtime_error("INT8 group 权重尺寸不匹配 w"+std::to_string(wid));graph.buffers.push_back(std::move(weights));auto*base=graph.buffers.back().data();std::vector<Qnn_Tensor_t>partial;for(unsigned g=0;g<k/group;g++){auto x=graph.slice(in[0],g*group,group);auto w=graph.tensor({group,n},1.0f/127,QNN_TENSOR_TYPE_APP_WRITE,nullptr,0,QNN_DATATYPE_SFIXED_POINT_8,0);w.v1.clientBuf={base+(size_t)g*group*n,group*n};graph.ins.push_back(w);auto raw=graph.tensor(shape,j["raw_step"]);graph.buffers.emplace_back(n*4,0);auto bias=graph.tensor({n},x.v1.quantizeParams.scaleOffsetEncoding.scale/127,QNN_TENSOR_TYPE_STATIC,graph.buffers.back().data(),n*4,QNN_DATATYPE_SFIXED_POINT_32,0);graph.op(QNN_OP_MAT_MUL,{x,w,bias},raw);std::vector<float>factors(n);for(unsigned c=0;c<n;c++)factors[c]=127*scales[g*n+c];auto factor=graph.constant(factors,{1,n});partial.push_back(graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,raw,factor,shape,j["partial_steps"][g]));}unsigned sum_id=0;while(partial.size()>1){std::vector<Qnn_Tensor_t>next;for(unsigned i=0;i<partial.size();i+=2){if(i+1==partial.size())next.push_back(partial[i]);else next.push_back(graph.binary(QNN_OP_ELEMENT_WISE_ADD,partial[i],partial[i+1],shape,j["sum_steps"][sum_id++]));}partial.swap(next);}graph.op(QNN_OP_CONVERT,{partial[0]},output);
   }else if(cfg.value("weight_format",std::string())=="per_channel_q4_residual"||cfg.value("weight_format",std::string())=="per_channel_i8_residual"){
    int wid=j["weight"];uint32_t k=in[0].v1.dimensions[1];unsigned bits=cfg.value("residual_bits",cfg.value("weight_format",std::string())=="per_channel_i8_residual"?8u:4u);unsigned residual_stages=cfg.value("residual_stages",2u);if(cfg.contains("single_stage_weights"))for(int single:cfg["single_stage_weights"])if(single==wid)residual_stages=1;if(residual_stages<1||residual_stages>2)throw std::runtime_error("residual_stages 只支持 1 或 2");std::vector<Qnn_Tensor_t>parts;for(unsigned stage=0;stage<residual_stages;stage++){std::string stem=bits==8?(stage==0?cfg.value("residual_base_stem",std::string(".pc8gguf")):cfg.value("residual_second_stem",std::string(".pc8ggufr1"))):".pc4r"+std::to_string(stage);auto weights=read<uint8_t>(root+"/w"+std::to_string(wid)+stem);auto scales=read<float>(root+"/w"+std::to_string(wid)+stem+"scales");if(weights.size()!=(size_t)k*n||scales.size()!=n)throw std::runtime_error("残差权重尺寸不匹配 w"+std::to_string(wid));graph.buffers.push_back(std::move(weights));auto weight=graph.lowbit({k,n},graph.buffers.back().data(),graph.buffers.back().size(),std::move(scales),bits);auto part=residual_stages==1?output:graph.tensor(shape,step);graph.op(QNN_OP_MAT_MUL,{in[0],weight},part);parts.push_back(part);}if(parts.size()==2)graph.op(QNN_OP_ELEMENT_WISE_ADD,parts,output);
   }else if(cfg.value("weight_format",std::string())=="custom_q4"){
    int wid=j["weight"];uint32_t k=in[0].v1.dimensions[1];auto weights=read<uint8_t>(root+"/w"+std::to_string(wid)+".q4hvx");auto scales=read<float>(root+"/w"+std::to_string(wid)+".q4hvxscales");
    if(weights.size()!=(size_t)k*n/2||scales.size()!=(size_t)(k/32)*n)throw std::runtime_error("自定义 Q4 权重尺寸不匹配 w"+std::to_string(wid));
    graph.buffers.push_back(std::move(weights));auto*code_base=graph.buffers.back().data();float scale_peak=*std::max_element(scales.begin(),scales.end());float scale_step=std::max(scale_peak/255.0f,1e-12f);graph.buffers.emplace_back(scales.size());auto*scale_base=graph.buffers.back().data();for(size_t i=0;i<scales.size();i++)scale_base[i]=std::clamp((int)std::round(scales[i]/scale_step),0,255);unsigned tiles=std::min<unsigned>(cfg.value("q4_tiles",1),n);unsigned tile_cols=(n+tiles-1)/tiles;std::vector<Qnn_Tensor_t>parts;
    for(unsigned start=0;start<n;start+=tile_cols){unsigned cols=std::min(tile_cols,n-start);auto codes=graph.tensor({cols,k/2},1,QNN_TENSOR_TYPE_STATIC,code_base+(size_t)start*k/2,(size_t)cols*k/2,QNN_DATATYPE_UFIXED_POINT_8,0);auto scale_tensor=graph.tensor({cols,k/32},scale_step,QNN_TENSOR_TYPE_STATIC,scale_base+(size_t)start*(k/32),(size_t)cols*(k/32),QNN_DATATYPE_UFIXED_POINT_8,0);auto part=tiles==1?output:graph.tensor({r,cols},step);graph.custom_op("ZllmQ4","BlockQ4Gemv",{in[0],codes,scale_tensor},part);parts.push_back(part);}
    if(parts.size()>1)graph.op(QNN_OP_CONCAT,parts,output,{graph.scalar("axis",1)});
   }else if(cfg.value("weight_format",std::string())=="per_channel_lowbit"){
    int wid=j["weight"],bits=cfg.value("weight_bits",6);if(cfg.contains("four_bit_weights"))for(int four:cfg["four_bit_weights"])if(four==wid)bits=4;uint32_t k=in[0].v1.dimensions[1];std::string stem=bits==4?cfg.value("four_bit_weight_stem",std::string(".pc4r0")):cfg.value("eight_bit_weight_stem",std::string(".pc"+std::to_string(bits)));if(cfg.contains("weight_stem"))stem=cfg["weight_stem"].get<std::string>();auto weights=read<uint8_t>(root+"/w"+std::to_string(wid)+stem);auto scales=read<float>(root+"/w"+std::to_string(wid)+stem+"scales");size_t weight_bytes=(size_t)k*n*(bits==16?2:1);if(weights.size()!=weight_bytes||scales.size()!=n)throw std::runtime_error("低比特权重尺寸不匹配 w"+std::to_string(wid));graph.buffers.push_back(std::move(weights));auto weight=graph.lowbit({k,n},graph.buffers.back().data(),graph.buffers.back().size(),std::move(scales),bits);graph.op(QNN_OP_MAT_MUL,{in[0],weight},output);
   }else if(cfg.value("weight_format",std::string())=="q4_grouped"){
    int wid=j["weight"];uint32_t k=in[0].v1.dimensions[1],groups=k/32;auto weights=read<uint8_t>(root+"/w"+std::to_string(wid)+".s4");auto scales=read<float>(root+"/w"+std::to_string(wid)+".q4scales");if(weights.size()!=(size_t)k*n||scales.size()!=(size_t)groups*n)throw std::runtime_error("分组 Q4 权重尺寸不匹配 w"+std::to_string(wid));graph.buffers.push_back(std::move(weights));auto*base=graph.buffers.back().data();std::vector<Qnn_Tensor_t>halves;
    for(unsigned g=0;g<groups;g++){auto x=graph.slice(in[0],g*32,32);std::vector<float>group_scales(scales.begin()+(size_t)g*n,scales.begin()+(size_t)(g+1)*n);auto weight=graph.lowbit({32,n},base+(size_t)g*32*n,32*n,std::move(group_scales),4);auto part=graph.tensor(shape,j["partial_steps"][g/2]);graph.op(QNN_OP_MAT_MUL,{x,weight},part);halves.push_back(part);}
    std::vector<Qnn_Tensor_t>partial;for(unsigned g=0;g<groups;g+=2)partial.push_back(graph.binary(QNN_OP_ELEMENT_WISE_ADD,halves[g],halves[g+1],shape,j["partial_steps"][g/2]));unsigned sum_id=0;while(partial.size()>1){std::vector<Qnn_Tensor_t>next;for(unsigned i=0;i<partial.size();i+=2){if(i+1==partial.size())next.push_back(partial[i]);else next.push_back(graph.binary(QNN_OP_ELEMENT_WISE_ADD,partial[i],partial[i+1],shape,j["sum_steps"][sum_id++]));}partial.swap(next);}graph.op(QNN_OP_CONVERT,{partial[0]},output);
   }else if(cfg.value("weight_format",std::string())=="q4_block32"){
    int wid=j["weight"];uint32_t k=in[0].v1.dimensions[1];auto weights=read<uint8_t>(root+"/w"+std::to_string(wid)+".s4");auto scales=read<float>(root+"/w"+std::to_string(wid)+".q4scales");
    if(weights.size()!=(size_t)k*n||scales.size()!=(size_t)(k/32)*n)throw std::runtime_error("Q4 权重尺寸不匹配 w"+std::to_string(wid));
    graph.buffers.push_back(std::move(weights));auto weight=graph.q4({k,n},graph.buffers.back().data(),graph.buffers.back().size(),scales,32);graph.op(QNN_OP_MAT_MUL,{in[0],weight},output);
   }else if(cfg.value("weight_format",std::string())=="global16"){
    int wid=j["weight"];uint32_t k=in[0].v1.dimensions[1];std::ifstream mf(root+"/w"+std::to_string(wid)+".global.json");if(!mf)throw std::runtime_error("缺少权重元数据 w"+std::to_string(wid));json wm;mf>>wm;uint32_t chunk=wm.value("chunk_columns",n);auto data=read<uint8_t>(root+"/w"+std::to_string(wid)+(chunk<n?".chunked16":".global16"));graph.buffers.push_back(std::move(data));auto*base=graph.buffers.back().data();std::vector<Qnn_Tensor_t>parts;uint64_t offset=0;
    for(unsigned start=0;start<n;start+=chunk){unsigned cols=std::min(chunk,n-start);auto w=graph.tensor({k,cols},wm["step"],QNN_TENSOR_TYPE_APP_WRITE,nullptr,0,QNN_DATATYPE_SFIXED_POINT_16,0);w.v1.clientBuf={base+offset,k*cols*2};offset+=(uint64_t)k*cols*2;graph.ins.push_back(w);auto part=chunk<n?graph.tensor({r,cols},step):output;graph.op(QNN_OP_MAT_MUL,{in[0],w},part);parts.push_back(part);}
    if(chunk<n)graph.op(QNN_OP_CONCAT,parts,output,{graph.scalar("axis",1)});

   }else{
   int wid=j["weight"];uint32_t k=in[0].v1.dimensions[1],group=k%64==0?64:k;bool wide=cfg.value("weight_bits",8)==16;unsigned bytes=wide?2:1;float level=wide?32767:127;auto weights=read<uint8_t>(root+"/w"+std::to_string(wid)+(wide?".i16":".i8"));auto scales=read<float>(root+"/w"+std::to_string(wid)+(wide?".scales16":".scales"));std::vector<Qnn_Tensor_t>partial;
   // 权重客户端缓冲必须活到 execute；存进 graph 自有字节缓冲。
   graph.buffers.emplace_back(weights.size());memcpy(graph.buffers.back().data(),weights.data(),weights.size());auto*base=graph.buffers.back().data();
   for(unsigned g=0;g<k/group;g++){auto x=graph.slice(in[0],g*group,group);bool fixed=cfg.value("static_weights",false);auto w=graph.tensor({group,n},1.0f/level,fixed?QNN_TENSOR_TYPE_STATIC:QNN_TENSOR_TYPE_APP_WRITE,fixed?base+g*group*n*bytes:nullptr,fixed?group*n*bytes:0,wide?QNN_DATATYPE_SFIXED_POINT_16:QNN_DATATYPE_SFIXED_POINT_8,0);w.v1.clientBuf={base+g*group*n*bytes,group*n*bytes};if(!fixed)graph.ins.push_back(w);auto raw=graph.tensor(shape,j["raw_step"]);graph.buffers.emplace_back(n*4,0);auto bias=graph.tensor({n},x.v1.quantizeParams.scaleOffsetEncoding.scale/level,QNN_TENSOR_TYPE_STATIC,graph.buffers.back().data(),n*4,QNN_DATATYPE_SFIXED_POINT_32,0);graph.op(QNN_OP_MAT_MUL,{x,w,bias},raw);std::vector<float>factors(n);for(unsigned c=0;c<n;c++)factors[c]=level*scales[g*n+c];auto factor=graph.constant(factors,{1,n});partial.push_back(graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,raw,factor,shape,j["partial_steps"][g]));}
   unsigned sum_id=0;while(partial.size()>1){std::vector<Qnn_Tensor_t>next;for(unsigned i=0;i<partial.size();i+=2){if(i+1==partial.size())next.push_back(partial[i]);else next.push_back(graph.binary(QNN_OP_ELEMENT_WISE_ADD,partial[i],partial[i+1],shape,j["sum_steps"][sum_id++]));}partial.swap(next);}graph.op(QNN_OP_CONVERT,{partial[0]},output);
  }
  }
  else if(op=="activation" && j["activation"]=="Silu"){
   auto sigmoid=graph.tensor(shape,1.0f/65535,QNN_TENSOR_TYPE_NATIVE,nullptr,0,QNN_DATATYPE_UFIXED_POINT_16,0);graph.op(QNN_OP_SIGMOID,{in[0]},sigmoid);
   float gate_step=std::max(meta[j["inputs"][0].get<int>()]["max"].get<float>()*1.1f/32767,1e-8f);
   auto silu=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,in[0],sigmoid,shape,gate_step);
   graph.op(QNN_OP_ELEMENT_WISE_MULTIPLY,{silu,in[1]},output);
  }
  else if(op=="activation"){
   if(j["activation"]!="GeluTanh")throw std::runtime_error("activation "+j["activation"].get<std::string>());
   auto lo=graph.binary(QNN_OP_ELEMENT_WISE_MAXIMUM,in[0],graph.constant(-5),shape,5.0f/32767);auto x=graph.binary(QNN_OP_ELEMENT_WISE_MINIMUM,lo,graph.constant(5),shape,5.0f/32767);auto x2=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,x,x,shape,25.0f/32767);auto x3=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,x2,x,shape,125.0f/32767);auto cube=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,x3,graph.constant(.044715f),shape,6.0f/32767);auto sum=graph.binary(QNN_OP_ELEMENT_WISE_ADD,x,cube,shape,11.0f/32767);auto arg=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,sum,graph.constant(.79788456f),shape,9.0f/32767);auto th=graph.tensor(shape,1.0f/32767);graph.op(QNN_OP_TANH,{arg},th);auto one=graph.binary(QNN_OP_ELEMENT_WISE_ADD,th,graph.constant(1),shape,2.0f/32767);auto half=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,one,graph.constant(.5f),shape,1.0f/32767);float gstep=meta[j["inputs"][0].get<int>()]["max"].get<float>()*1.1f/32767;auto gelu=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,in[0],half,shape,gstep);graph.op(QNN_OP_ELEMENT_WISE_MULTIPLY,{gelu,in[1]},output);
  }
  else if(op=="rope"){
   uint32_t heads=j["heads"],dim=j["dim"],hd=n/heads,half=dim/2,position=j["position"];auto x=graph.reshape(in[0],{r*heads,hd});std::string table=j["table"];std::vector<float>cos,sin;if(!chat){cos=read<float>(root+"/"+table+"_cos.f32");sin=read<float>(root+"/"+table+"_sin.f32");}std::vector<float> permutation((size_t)hd*hd,0),cs((size_t)r*heads*hd,1),sn(cs.size(),0);
   bool interleaved=j.value("layout",std::string())=="Interleaved";float rope_theta=cfg.value("rope_theta",5000000.0f);for(unsigned d=0;d<half;d++){unsigned a=interleaved?2*d:d,b=interleaved?2*d+1:d+half;permutation[(size_t)b*hd+a]=-1;permutation[(size_t)a*hd+b]=1;for(unsigned t=0;t<r;t++)for(unsigned h=0;h<heads;h++){size_t base=((size_t)t*heads+h)*hd;float c,s;if(chat){float angle=(position+t)/std::pow(rope_theta,2.0f*d/dim);c=std::cos(angle);s=std::sin(angle);}else{c=cos[(position+t)*half+d];s=sin[(position+t)*half+d];}cs[base+a]=cs[base+b]=c;sn[base+a]=sn[base+b]=s;}}
   float xstep=in[0].v1.quantizeParams.scaleOffsetEncoding.scale;auto perm=graph.constant(permutation,{hd,hd});auto rotated=graph.tensor({r*heads,hd},xstep);graph.op(QNN_OP_MAT_MUL,{x,perm},rotated);auto c=graph.constant(cs,{r*heads,hd},1.0f/32767),s=graph.constant(sn,{r*heads,hd},1.0f/32767);auto real=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,x,c,{r*heads,hd},xstep),imaginary=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,rotated,s,{r*heads,hd},xstep);auto sum=graph.binary(QNN_OP_ELEMENT_WISE_ADD,real,imaginary,{r*heads,hd},step);auto flat=graph.reshape(sum,shape);graph.op(QNN_OP_CONVERT,{flat},output);
  }
  else if(op=="attention"||op=="attention_shared"){
   unsigned layer=j["layer"],heads=j["heads"],kvheads=j["kv_heads"],d=j["dim"],position=j["position"],window=j["window"];float score_scale=j.value("score_scale",1.0f);Qnn_Tensor_t key,value,key_export{},value_export{};unsigned length=position+r;
   if(op=="attention"){
    float ks=in[1].v1.quantizeParams.scaleOffsetEncoding.scale,vs=in[2].v1.quantizeParams.scaleOffsetEncoding.scale;
    if(position){if(!cache.count(layer))throw std::runtime_error("缺少前序 KV");ks=std::max(ks,cache[layer].first->step);vs=std::max(vs,cache[layer].second->step);}
    next_k=std::make_unique<Value>(Value{std::vector<uint16_t>(length*kvheads*d),length,kvheads*d,ks});next_v=std::make_unique<Value>(Value{std::vector<uint16_t>(length*kvheads*d),length,kvheads*d,vs});if(fused){key=graph.tensor({length,kvheads*d},ks);value=graph.tensor({length,kvheads*d},vs);key_export=graph.output(*next_k);value_export=graph.output(*next_v);}else{key=graph.output(*next_k);value=graph.output(*next_v);}next_layer=layer;
    if(position){auto old_k=graph.input(*cache[layer].first),old_v=graph.input(*cache[layer].second);auto ok=graph.tensor({position,kvheads*d},ks),nk=graph.tensor({r,kvheads*d},ks),ov=graph.tensor({position,kvheads*d},vs),nv=graph.tensor({r,kvheads*d},vs);graph.op(QNN_OP_CONVERT,{old_k},ok);graph.op(QNN_OP_CONVERT,{in[1]},nk);graph.op(QNN_OP_CONVERT,{old_v},ov);graph.op(QNN_OP_CONVERT,{in[2]},nv);graph.op(QNN_OP_CONCAT,{ok,nk},key,{graph.scalar("axis",0)});graph.op(QNN_OP_CONCAT,{ov,nv},value,{graph.scalar("axis",0)});}else{graph.op(QNN_OP_CONVERT,{in[1]},key);graph.op(QNN_OP_CONVERT,{in[2]},value);}
   }else{key=graph.input(*cache.at(layer).first);value=graph.input(*cache.at(layer).second);length=cache.at(layer).first->rows;}
   if(fused&&op=="attention"){graph.op(QNN_OP_CONVERT,{key},key_export);graph.op(QNN_OP_CONVERT,{value},value_export);}
   float score_max=j.value("score_max",256.0f/score_scale),mask_value=-score_max-32/score_scale;std::vector<uint8_t>mask(r*length);for(unsigned t=0;t<r;t++)for(unsigned k=0;k<length;k++)if(k>position+t||(window&&k+window<=position+t))mask[t*length+k]=1;
   graph.buffers.push_back(std::move(mask));auto mask_t=graph.tensor({r,length},0,QNN_TENSOR_TYPE_STATIC,graph.buffers.back().data(),graph.buffers.back().size(),QNN_DATATYPE_BOOL_8);std::vector<Qnn_Tensor_t>outputs;
   for(unsigned h=0;h<heads;h++){auto q=graph.slice(in[0],h*d,d);auto k=graph.slice(key,(h/(heads/kvheads))*d,d);auto v=graph.slice(value,(h/(heads/kvheads))*d,d);auto kt=graph.tensor({d,length},key.v1.quantizeParams.scaleOffsetEncoding.scale);graph.op(QNN_OP_TRANSPOSE,{k},kt,{graph.array("perm",{1,0})});auto scores=graph.tensor({r,length},score_max/32767);graph.op(QNN_OP_MAT_MUL,{q,kt},scores);auto masked=graph.tensor({r,length},-mask_value/32767);graph.op(QNN_OP_ELEMENT_WISE_SELECT,{mask_t,graph.constant(mask_value),scores},masked);auto prob=graph.tensor({r,length},1.0f/65535,QNN_TENSOR_TYPE_NATIVE,nullptr,0,QNN_DATATYPE_UFIXED_POINT_16,0);auto beta=graph.scalar("beta",0,QNN_DATATYPE_FLOAT_32);beta.scalarParam.floatValue=score_scale;graph.op(QNN_OP_SOFTMAX,{masked},prob,{graph.scalar("axis",1),beta});auto head=graph.tensor({r,d},step);graph.op(QNN_OP_MAT_MUL,{prob,v},head);outputs.push_back(head);}
   graph.op(QNN_OP_CONCAT,outputs,output,{graph.scalar("axis",1)});
  }
  else if(op=="relu"){
   graph.op(QNN_OP_RELU,{in[0]},output);
  }
  else if(op=="row_bias"){
   auto bias=read<float>(root+"/w"+std::to_string(j["bias"].get<int>())+".f32");
   if(bias.size()!=(size_t)n)throw std::runtime_error("row_bias 尺寸不匹配");
   // 广播展开为 r×n 常量，回避 elementwise 广播支持差异
   std::vector<float> expanded((size_t)r*n);for(unsigned t=0;t<r;t++)std::copy(bias.begin(),bias.end(),expanded.begin()+(size_t)t*n);
   graph.op(QNN_OP_ELEMENT_WISE_ADD,{in[0],graph.constant(expanded,shape)},output);
  }
  else if(op=="layernorm"){
   auto gamma=read<float>(root+"/w"+std::to_string(j["weight"].get<int>())+".f32"),beta=read<float>(root+"/w"+std::to_string(j["bias"].get<int>())+".f32");
   if(gamma.size()!=(size_t)n||beta.size()!=(size_t)n)throw std::runtime_error("layernorm 权重尺寸不匹配");
   float xs=in[0].v1.quantizeParams.scaleOffsetEncoding.scale;
   float xmax=std::max(meta.at(j["inputs"][0].get<int>())["max"].get<float>(),1e-6f);
   // 均值/中心化走 MATMUL×全1（只依赖已验证的 [1,n]/[1,1] 广播）；
   // 分母用与 rmsnorm 相同的 concat-sqrt(n*eps) + L2_NORM 分解，sqrt(n) 折入 gamma。
   // 行和的动态范围远小于 n*xmax 上界，用离线参考实测校准，避免量化步长过粗。
   auto reference_in=read<float>(root+"/t"+std::to_string(j["inputs"][0].get<int>())+".f32");
   float sum_max=xmax,centered_max=xmax;for(unsigned t=0;t<r&&t*n<reference_in.size();t++){float row=0;for(unsigned c=0;c<n;c++)row+=reference_in[(size_t)t*n+c];sum_max=std::max(sum_max,std::abs(row));centered_max=std::max(centered_max,std::abs(row)/(float)n+xmax);}
   sum_max*=1.1f;centered_max*=1.1f;
   std::vector<float> ones_n(n,1.0f);
   auto sum=graph.tensor({r,1},std::max(sum_max/32767.0f,1e-9f));graph.op(QNN_OP_MAT_MUL,{in[0],graph.constant(ones_n,{n,1})},sum);
   auto mean=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,sum,graph.constant(1.0f/(float)n),{r,1},std::max(sum_max/(float)n/32767.0f,1e-9f));
   auto neg_mean=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,mean,graph.constant(-1.0f),{r,1},std::max(sum_max/(float)n/32767.0f,1e-9f));
   auto shift=graph.tensor(shape,std::max(centered_max/32767.0f,xs));graph.op(QNN_OP_MAT_MUL,{neg_mean,graph.constant(ones_n,{1,n})},shift);
   auto centered=graph.tensor(shape,xs);graph.op(QNN_OP_ELEMENT_WISE_ADD,{in[0],shift},centered);
   auto epsilon=graph.constant(std::vector<float>(r,std::sqrt((float)n*j["eps"].get<float>())),{r,1},xs);
   auto eps_same=graph.tensor({r,1},xs);graph.op(QNN_OP_CONVERT,{epsilon},eps_same);
   auto padded=graph.tensor({r,n+1},xs);graph.op(QNN_OP_CONCAT,{centered,eps_same},padded,{graph.scalar("axis",1)});
   auto unit=graph.tensor({r,n+1},1.1f/32767.0f);auto l2eps=graph.scalar("epsilon",0,QNN_DATATYPE_FLOAT_32);l2eps.scalarParam.floatValue=1e-12f;
   graph.op(QNN_OP_L2_NORM,{padded},unit,{graph.scalar("axis",1),l2eps});
   auto normalized=graph.slice(unit,0,n);
   std::vector<float> scaled_gamma(n);for(unsigned c=0;c<n;c++)scaled_gamma[c]=gamma[c]*std::sqrt((float)n);
   auto scaled=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,normalized,graph.constant(scaled_gamma,{1,n}),shape,step);
   graph.op(QNN_OP_ELEMENT_WISE_ADD,{scaled,graph.constant(beta,{1,n})},output);
  }
  else if(op=="full_attention"){
   unsigned heads=j["heads"],d=j["dim"];float scale=j["scale"];
   // score 范围从离线参考精确标定（q@k^T 逐头最大值），避免 step 过粗削顶或过细溢出
   auto q_ref=read<float>(root+"/t"+std::to_string(j["inputs"][0].get<int>())+".f32");
   auto k_ref=read<float>(root+"/t"+std::to_string(j["inputs"][1].get<int>())+".f32");
   float score_max=1.0f;
   for(unsigned row=0;row<r;row++)for(unsigned head=0;head<heads;head++){
    const float*q=&q_ref[(size_t)row*n+head*d];float peak=0;
    for(unsigned other=0;other<r;other++){
     const float*k=&k_ref[(size_t)other*n+head*d];float dot=0;
     for(unsigned e=0;e<d;e++)dot+=q[e]*k[e];
     peak=std::max(peak,std::abs(dot));
    }
    score_max=std::max(score_max,peak);
   }
   score_max*=1.1f;
   std::vector<Qnn_Tensor_t> outputs;
   for(unsigned h=0;h<heads;h++){
    auto q=graph.slice(in[0],h*d,d),k=graph.slice(in[1],h*d,d),v=graph.slice(in[2],h*d,d);
    auto kt=graph.tensor({d,r},in[1].v1.quantizeParams.scaleOffsetEncoding.scale);graph.op(QNN_OP_TRANSPOSE,{k},kt,{graph.array("perm",{1,0})});
    auto scores=graph.tensor({r,r},score_max/32767.0f);graph.op(QNN_OP_MAT_MUL,{q,kt},scores);
    auto prob=graph.tensor({r,r},1.0f/65535,QNN_TENSOR_TYPE_NATIVE,nullptr,0,QNN_DATATYPE_UFIXED_POINT_16,0);
    auto beta=graph.scalar("beta",0,QNN_DATATYPE_FLOAT_32);beta.scalarParam.floatValue=scale;
    graph.op(QNN_OP_SOFTMAX,{scores},prob,{graph.scalar("axis",1),beta});
    auto head=graph.tensor({r,d},step);graph.op(QNN_OP_MAT_MUL,{prob,v},head);outputs.push_back(head);
   }
   graph.op(QNN_OP_CONCAT,outputs,output,{graph.scalar("axis",1)});
  }
  else if(op=="depthwise"){
   unsigned kernel=j["kernel"],left=j["left"],right=j["right"];
   auto weights=read<float>(root+"/w"+std::to_string(j["weight"].get<int>())+".f32");
   if(weights.size()!=(size_t)kernel*n)throw std::runtime_error("depthwise 权重尺寸不匹配");
   float xs=in[0].v1.quantizeParams.scaleOffsetEncoding.scale;
   // 零填充展开（CONCAT 前统一量化编码），再按 tap 做 移位切片×权重 累加，
   // 回避 HTP 分组卷积支持差异；kernel=11 时只有 11 次 MUL + 10 次 ADD。
   Qnn_Tensor_t padded=in[0];
   if(left||right){
    std::vector<Qnn_Tensor_t> parts;
    if(left){auto zero_l=graph.tensor({left,n},xs);graph.op(QNN_OP_CONVERT,{graph.constant(std::vector<float>((size_t)left*n,0.0f),{left,n})},zero_l);parts.push_back(zero_l);}
    parts.push_back(in[0]);
    if(right){auto zero_r=graph.tensor({right,n},xs);graph.op(QNN_OP_CONVERT,{graph.constant(std::vector<float>((size_t)right*n,0.0f),{right,n})},zero_r);parts.push_back(zero_r);}
    padded=graph.tensor({r+left+right,n},xs);graph.op(QNN_OP_CONCAT,parts,padded,{graph.scalar("axis",0)});
   }
   // product/累加步长从离线参考精确标定：逐 tap 统计 |x_shift×w| 与部分和最大值
   auto conv_reference=read<float>(root+"/t"+std::to_string(j["inputs"][0].get<int>())+".f32");
   std::vector<float> padded_reference((size_t)(r+left+right)*n,0.0f);
   std::copy(conv_reference.begin(),conv_reference.end(),padded_reference.begin()+(size_t)left*n);
   float product_peak=1e-12f,accumulate_peak=1e-12f;
   for(unsigned row=0;row<r;row++){
    std::vector<float> accumulated_row(n,0.0f);
    for(unsigned tap=0;tap<kernel;tap++){
     float tap_peak=0;const float*source=&padded_reference[((size_t)row+tap)*n];const float*w=&weights[(size_t)tap*n];
     for(unsigned c=0;c<n;c++){float product=source[c]*w[c];tap_peak=std::max(tap_peak,std::abs(product));accumulated_row[c]+=product;}
     product_peak=std::max(product_peak,tap_peak);
    }
    for(float v:accumulated_row)accumulate_peak=std::max(accumulate_peak,std::abs(v));
   }
   float product_step=std::max(product_peak*1.1f/32767.0f,1e-12f);
   float accumulate_step=std::max(accumulate_peak*1.1f/32767.0f,1e-12f);
   Qnn_Tensor_t accumulated{};
   for(unsigned tap=0;tap<kernel;tap++){
    auto shifted=graph.slice(padded,tap,r,0);
    auto weight=graph.constant(std::vector<float>(weights.begin()+(size_t)tap*n,weights.begin()+(size_t)(tap+1)*n),{1,n});
    auto product=graph.binary(QNN_OP_ELEMENT_WISE_MULTIPLY,shifted,weight,shape,product_step);
    accumulated=tap==0?product:graph.binary(QNN_OP_ELEMENT_WISE_ADD,accumulated,product,shape,accumulate_step);
   }
   graph.op(QNN_OP_CONVERT,{accumulated},output);
  }
  else throw std::runtime_error("未实现 HTP 算子 "+op+" id="+std::to_string(id));
  if(fused){fused_tensors[id]=output;if(next_layer>=0)fused_cache.emplace_back(next_layer,std::move(next_k),std::move(next_v));if(id%100==0){printf("BUILD ops=%d\n",id);fflush(stdout);}if(seg_end)close_segment();continue;}
  graph.execute(cfg.value("repeats",0));if(cfg.contains("dump_dir"))write(cfg["dump_dir"].get<std::string>()+"/t"+std::to_string(id)+".u16",out->data);if(next_layer>=0)cache[next_layer]={std::move(next_k),std::move(next_v)};double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  // 数值审计仅在执行后进行，不替换输出，不把参考数据馈入后续算子。
  float maxerr=0;unsigned bad=0;if(audit){auto reference=read<float>(audit_root+"/t"+std::to_string(id)+".f32");for(unsigned i=0;i<out->data.size();i++){float actual=((int)out->data[i]-32768)*step;float e=std::abs(actual-reference[i]);maxerr=std::max(maxerr,e);bad+=e>audit_atol+audit_rtol*std::abs(reference[i]);}}printf("stage=%d op=%s shape=%ux%u ms=%.2f maxerr=%.6f bad=%u\n",id,op.c_str(),r,n,ms,maxerr,bad);if(bad&&cfg.value("print_mismatch",false)){for(unsigned i=0;i<std::min<size_t>(8,out->data.size());i++)printf("sample=%u q=%u actual=%.6f\n",i,out->data[i],((int)out->data[i]-32768)*step);}fflush(stdout);failures+=bad!=0;stages++;values[id]=std::move(out);if(bad&&cfg.value("stop_on_error",true))throw std::runtime_error("数值门禁失败 id="+std::to_string(id));
 }
 if(fused){
  close_segment();
  // wall = 全部分段构建+finalize+执行+host 解码的总耗时
  double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fused_begin).count();
  auto expected_path=root+"/expected.json";std::ifstream ef(expected_path);if(!ef)throw std::runtime_error("缺少 expected.json");json expected;ef>>expected;
  auto &logits=*values.at(final_id);
  uint32_t rows=logits.rows,cols=logits.cols;
  std::vector<uint32_t> frame_ids;for(unsigned t=0;t<rows;t++){unsigned best=0;float best_value=-1e30f;for(unsigned c=0;c<cols;c++){float v=((int)logits.data[(size_t)t*cols+c]-32768)*logits.step;if(v>best_value){best_value=v;best=c;}}frame_ids.push_back(best);}
  std::vector<uint32_t> reference_ids=expected["frame_ids"];
  unsigned matched=0;for(size_t i=0;i<frame_ids.size()&&i<reference_ids.size();i++)matched+=frame_ids[i]==reference_ids[i];
  // CTC 贪心：合并连续重复 → 去 blank → 跳过 4 个控制位
  std::vector<uint32_t> tokens;for(size_t i=0;i<frame_ids.size();i++)if(!i||frame_ids[i]!=frame_ids[i-1])if(frame_ids[i]!=0)tokens.push_back(frame_ids[i]);
  if(tokens.size()>4)tokens.erase(tokens.begin(),tokens.begin()+4);else tokens.clear();
  std::map<uint32_t,std::string> vocabulary;{std::ifstream tf(cfg.at("tokens").get<std::string>());std::string line;while(std::getline(tf,line)){auto at=line.rfind(' ');if(at==std::string::npos)continue;try{vocabulary[std::stoul(line.substr(at+1))]=line.substr(0,at);}catch(...){}}}
  std::string text;for(uint32_t t:tokens)text+=vocabulary.count(t)?vocabulary[t]:"";
  printf("ASR_FRAMES total=%zu matched=%u\n",frame_ids.size(),matched);
  printf("SENSEVOICE frames=%zu wall_ms=%.3f text=%s\n",frame_ids.size(),wall,text.c_str());fflush(stdout);
  return frame_ids.size()==reference_ids.size()&&matched==frame_ids.size()?0:1;
 }
 printf("stages=%u failed=%u\n",stages,failures);return failures?1:0;
 }catch(std::exception&e){fprintf(stderr,"ERROR: %s\n",e.what());return 2;}}

