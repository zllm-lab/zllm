// App 常驻 LLM 引擎 runner(原 zllm-bench qnn_htp_cached_decode):加载离线
// context binary 执行 decode/prefill,KV checkpoint/恢复、固定前缀缓存、AR8
// 批量 prefill、DSpark MTP 与 --server 常驻复用都在这里。模型维度
// (model_layers/KV capacity/trace 元数据 id)全部来自请求配置,不绑定具体模型。
// 由 zllm-qnn-decode bin 经 C ABI 调用,stdout 协议与 App JNI 保持兼容。
#include "htp_graph.hpp"
#include "htp_session.hpp"
#include <HTP/QnnHtpContext.h>
#include <HTP/QnnHtpMem.h>
#include <HTP/QnnHtpSystemContext.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <csignal>

static volatile std::sig_atomic_t checkpoint_requested=0;
static volatile std::sig_atomic_t cancel_requested=0;
static void request_checkpoint(int signal){checkpoint_requested=1;if(signal==SIGTERM)cancel_requested=1;}

static uint64_t spill_size(const QnnSystemContext_BinaryInfo_t* binary){
 if(binary->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3){
  auto&context=binary->contextBinaryInfoV3;if(context.numGraphs!=1)throw std::runtime_error("共享临时缓冲需要单图 context");auto&g=context.graphs[0];
  if(g.version!=QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3||g.graphInfoV3.graphBlobInfoSize<sizeof(QnnHtpSystemContext_GraphBlobInfo_t))throw std::runtime_error("缺少 HTP 临时缓冲大小元数据");
  auto*blob=static_cast<QnnHtpSystemContext_GraphBlobInfo_t*>(g.graphInfoV3.graphBlobInfo);if(blob->version!=QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1)throw std::runtime_error("不支持的 HTP graph blob 版本");return blob->contextBinaryGraphBlobInfoV1.spillFillBufferSize;
 }
 void*data=binary->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1?binary->contextBinaryInfoV1.hwInfoBlob:binary->contextBinaryInfoV2.hwInfoBlob;
 if(!data)throw std::runtime_error("缺少 HTP hardware blob");auto*blob=static_cast<QnnHtpSystemContext_HwBlobInfo_t*>(data);if(blob->version!=QNN_SYSTEM_CONTEXT_HTP_HW_INFO_BLOB_VERSION_V1)throw std::runtime_error("不支持的 HTP hardware blob 版本");return blob->contextBinaryHwInfoBlobV1_t.spillFillBufferSize;
}

struct CachedGraph {
 QnnSystemContext_Handle_t system=nullptr;
 Qnn_ContextHandle_t context=nullptr;
 Qnn_GraphHandle_t graph=nullptr;
 std::vector<Qnn_Tensor_t> inputs,outputs;
};

struct ScopeExit {
 std::function<void()> release;
 ~ScopeExit(){if(release)release();}
};

struct SharedArena {
 void* data=nullptr;
 int fd=-1;
 size_t size=0,used=0;
 void (*release)(void*)=nullptr;
 ~SharedArena(){if(data)release(data);}
};

struct SharedTensor {
 QNN_INTERFACE_VER_TYPE* api=nullptr;
 Qnn_MemHandle_t handle=nullptr;
 void* data=nullptr;
 void (*release)(void*)=nullptr;
 std::shared_ptr<SharedArena> arena;
 ~SharedTensor(){if(handle)api->memDeRegister(&handle,1);if(data&&release)release(data);}
};

struct LayerBuffers {
 std::vector<uint16_t> hidden_in,hidden_out,key_in,value_in,key_out,value_out;
 std::unique_ptr<SharedTensor> shared_key,shared_value;
 uint16_t* key(){return shared_key?static_cast<uint16_t*>(shared_key->data):key_in.data();}
 uint16_t* value(){return shared_value?static_cast<uint16_t*>(shared_value->data):value_in.data();}
};

static void require_same_encoding(const Qnn_Tensor_t&source,const Qnn_Tensor_t&target,const std::string&name){
 const auto&a=source.v1;const auto&b=target.v1;
 if(a.dataType!=b.dataType||a.quantizeParams.quantizationEncoding!=b.quantizeParams.quantizationEncoding||b.quantizeParams.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET||a.quantizeParams.scaleOffsetEncoding.scale!=b.quantizeParams.scaleOffsetEncoding.scale||a.quantizeParams.scaleOffsetEncoding.offset!=b.quantizeParams.scaleOffsetEncoding.offset)throw std::runtime_error(name+" 跨图量化编码不一致，禁止直接复制数值 buffer");
}

#include "qnn_dspark_runtime.hpp"

extern "C" int zllm_qnn_decode_main(int argc,char**argv){try{
 std::signal(SIGUSR1,request_checkpoint);std::signal(SIGTERM,request_checkpoint);
 bool server=argc==3&&std::string(argv[1])=="--server";
 if(!(server||(argc==3&&std::string(argv[1])=="--config")))return 2;std::ifstream cf(argv[2]);json cfg;cf>>cfg;
 // 不兼容的缓存在加载 HTP context 前拒绝，避免带着未释放的厂商资源退出。
 {
  bool fresh=cfg.contains("prompt_tokens"),restore_prefix=fresh&&cfg.contains("prefix_state_dir");std::vector<uint32_t> prefix_tokens;json state;
  if(restore_prefix){prefix_tokens=cfg.at("prefix_tokens").get<std::vector<uint32_t>>();auto supplied=cfg.at("prompt_tokens").get<std::vector<uint32_t>>();if(prefix_tokens.empty()||supplied.size()<=prefix_tokens.size()||!std::equal(prefix_tokens.begin(),prefix_tokens.end(),supplied.begin()))throw std::runtime_error("缓存前缀与当前 prompt 不一致");}
  if(!fresh){std::string initial=restore_prefix?cfg.at("prefix_state_dir").get<std::string>():cfg.at("state_dir").get<std::string>();std::ifstream sf(initial+"/state.json");if(!sf)throw std::runtime_error("缺少 prefill state");}
 }
 auto session=htp_session_open(cfg,true);
 auto&api=session.api;auto&system=session.system;Qnn_BackendHandle_t backend=session.backend;Qnn_DeviceHandle_t device=session.device;
 if(cfg.value("slice_probe",false)){
  for(auto spec:std::vector<std::vector<unsigned>>{{3,7,0,1,2},{3,7,1,2,4},{4096,256,1,128,128}}){unsigned rows=spec[0],cols=spec[1],axis=spec[2],start=spec[3],count=spec[4];Value input{std::vector<uint16_t>((size_t)rows*cols),rows,cols,0.01f},output{std::vector<uint16_t>((size_t)(axis==0?count:rows)*(axis==1?count:cols)),axis==0?count:rows,axis==1?count:cols,0.01f};for(size_t i=0;i<input.data.size();i++)input.data[i]=30000+i%4000;{Graph g(api,backend,device);auto result=g.slice(g.input(input),start,count,axis);g.op(QNN_OP_CONVERT,{result},g.output(output));g.execute();}for(unsigned r=0;r<output.rows;r++)for(unsigned c=0;c<output.cols;c++)if(output.data[(size_t)r*output.cols+c]!=input.data[(size_t)(r+(axis==0?start:0))*cols+c+(axis==1?start:0)])throw std::runtime_error("StridedSlice 与参考结果不一致");printf("SLICE_PASS rows=%u cols=%u axis=%u\n",rows,cols,axis);}
  return 0;
 }
 // 与 Genie 相同，通过注册 RPC buffer 复用 KV 内存，避免每层每轮重新搬运整块缓存。
 std::shared_ptr<SharedArena> kv_arena;unsigned shared_count=0;
 auto share=[&](Qnn_ContextHandle_t context,Qnn_Tensor_t&tensor,const void*source,size_t size){
  void*library=dlopen("libcdsprpc.so",RTLD_NOW);if(!library)throw std::runtime_error("无法加载 RPC allocator");auto allocate=reinterpret_cast<void*(*)(int,uint32_t,int)>(dlsym(library,"rpcmem_alloc"));auto to_fd=reinterpret_cast<int(*)(void*)>(dlsym(library,"rpcmem_to_fd"));auto release=reinterpret_cast<void(*)(void*)>(dlsym(library,"rpcmem_free"));if(!allocate||!to_fd||!release)throw std::runtime_error("RPC allocator API 缺失");
  auto memory=std::make_unique<SharedTensor>();memory->api=&api;Qnn_MemDescriptor_t descriptor=QNN_MEM_DESCRIPTOR_INIT;descriptor.memShape={tensor.v1.rank,tensor.v1.dimensions,nullptr};descriptor.dataType=tensor.v1.dataType;QnnMemHtp_Descriptor_t custom{};
  if(cfg.value("shared_kv_arena",false)){
   size_t aligned=(size+4095)&~size_t(4095);
   if(!kv_arena){kv_arena=std::make_shared<SharedArena>();kv_arena->size=aligned*2*cfg.value("model_layers",42u);if(kv_arena->size>INT32_MAX)throw std::runtime_error("共享 KV arena 超出 RPC allocator 容量");kv_arena->release=release;kv_arena->data=allocate(25,1,kv_arena->size);if(!kv_arena->data)throw std::runtime_error("RPC KV arena 分配失败 bytes="+std::to_string(kv_arena->size));kv_arena->fd=to_fd(kv_arena->data);}
   if(kv_arena->used+aligned>kv_arena->size)throw std::runtime_error("共享 KV arena 越界");memory->arena=kv_arena;memory->data=static_cast<char*>(kv_arena->data)+kv_arena->used;custom.type=QNN_HTP_MEM_SHARED_BUFFER;custom.size=kv_arena->size;custom.sharedBufferConfig={kv_arena->fd,kv_arena->used};descriptor.memType=QNN_MEM_TYPE_CUSTOM;descriptor.customInfo=&custom;kv_arena->used+=aligned;
  }else{memory->release=release;memory->data=allocate(25,1,size);if(!memory->data)throw std::runtime_error("RPC KV 分配失败 bytes="+std::to_string(size));descriptor.memType=QNN_MEM_TYPE_ION;descriptor.ionInfo.fd=to_fd(memory->data);}
  uint32_t flat_elements=size/2;if(cfg.value("shared_kv_flat",false))descriptor.memShape={1,&flat_elements,nullptr};
  int fd=memory->arena?memory->arena->fd:descriptor.ionInfo.fd;if(fd<0)throw std::runtime_error("RPC KV fd 无效 bytes="+std::to_string(size));
  memcpy(memory->data,source,size);auto rc=api.memRegister(context,&descriptor,1,&memory->handle);if(rc)throw std::runtime_error("register KV memory index="+std::to_string(shared_count)+" bytes="+std::to_string(size)+" fd="+std::to_string(fd)+" status="+std::to_string(rc));shared_count++;tensor.v1.memType=QNN_TENSORMEMTYPE_MEMHANDLE;tensor.v1.memHandle=memory->handle;return memory;
 };
 std::map<int,json>meta;std::ifstream tf(cfg.at("root").get<std::string>()+"/trace.jsonl");std::string line;while(std::getline(tf,line)){auto j=json::parse(line);if(j.contains("id"))meta[j["id"].get<int>()]=j;}
 unsigned layers=cfg.value("model_layers",42u),capacity=cfg.value("resident_kv_capacity",4096u);auto stop_tokens=cfg.value("stop_tokens",std::vector<uint32_t>{1,130073});unsigned hidden_size=meta.at(cfg.value("meta_hidden_id",634))["cols"],head_dim=meta.at(cfg.value("meta_attention_id",641))["dim"],kv_cols=meta.at(cfg.value("meta_attention_id",641))["kv_heads"].get<unsigned>()*head_dim,vocab=meta.at(cfg.value("meta_head_id",1266))["cols"];
 Qnn_ContextHandle_t shared_group=nullptr;uint64_t max_spill=0;
 // 上下文 mount 循环:按当前请求装载图集;容量用满时 checkpoint 后换 ladder
 // 下一档重载,从 checkpoint 恢复继续生成,避免一开始就按大上下文占满内存。
 auto load=[&](const std::string&path,uint32_t expected_inputs,uint32_t expected_outputs){if(cancel_requested)throw std::runtime_error("context 加载已取消");CachedGraph result;ScopeExit failed{[&]{if(result.context)api.contextFree(result.context,nullptr);if(result.system)system.systemContextFree(result.system);}};auto bytes=read<uint8_t>(path);check(system.systemContextCreate(&result.system),"system context");const QnnSystemContext_BinaryInfo_t*binary=nullptr;Qnn_ContextBinarySize_t size=0;check(system.systemContextGetBinaryInfo(result.system,bytes.data(),bytes.size(),&binary,&size),"binary info");auto info=graph_info(binary);if(info.input_count!=expected_inputs||info.output_count!=expected_outputs)throw std::runtime_error("context 输入输出数量不匹配: "+path);result.inputs.assign(info.inputs,info.inputs+info.input_count);result.outputs.assign(info.outputs,info.outputs+info.output_count);QnnHtpContext_CustomConfig_t io=QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;io.option=QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;io.ioMemEstimation=true;QnnContext_Config_t option=QNN_CONTEXT_CONFIG_INIT;option.option=QNN_CONTEXT_CONFIG_OPTION_CUSTOM;option.customConfig=&io;QnnHtpContext_CustomConfig_t group=QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;group.option=QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;group.groupRegistration={shared_group,max_spill};QnnContext_Config_t group_option=QNN_CONTEXT_CONFIG_INIT;group_option.option=QNN_CONTEXT_CONFIG_OPTION_CUSTOM;group_option.customConfig=&group;std::vector<const QnnContext_Config_t*>options;if(cfg.value("context_io_estimation",false))options.push_back(&option);bool share_group=cfg.value("share_spill_fill",false)&&(!cfg.contains("dspark_cache_dir")||path.rfind(cfg.at("dspark_cache_dir").get<std::string>()+"/",0)!=0);if(share_group)options.push_back(&group_option);options.push_back(nullptr);check(api.contextCreateFromBinary(backend,device,options.data(),bytes.data(),bytes.size(),&result.context,nullptr),"load context");if(share_group&&!shared_group)shared_group=result.context;check(api.graphRetrieve(result.context,info.name,&result.graph),"retrieve graph");failed.release={};return result;};
 if(cfg.contains("embedding_probe_tokens")){
  for(uint32_t wanted:cfg["embedding_probe_tokens"]){uint32_t part=wanted/(32*51),group=(wanted/32)%51,row=wanted%32;auto embedding=load(cfg.at("context_cache_dir").get<std::string>()+"/embedding_"+std::to_string(part)+".bin",2,1);std::vector<uint16_t> out(hidden_size);embedding.inputs[0].v1.clientBuf={&group,4};embedding.inputs[1].v1.clientBuf={&row,4};embedding.outputs[0].v1.clientBuf={out.data(),uint32_t(out.size()*2)};auto rc=api.graphExecute(embedding.graph,embedding.inputs.data(),2,embedding.outputs.data(),1,nullptr,nullptr);printf("EMBEDDING_PROBE token=%u part=%u group=%u row=%u status=%llu\n",wanted,part,group,row,(unsigned long long)rc);fflush(stdout);api.contextFree(embedding.context,nullptr);system.systemContextFree(embedding.system);if(rc)return 2;}
  return 0;
 }
 if(cfg.contains("profile_graph")){
  auto g=load(cfg.at("profile_graph"),6,3);std::vector<std::vector<uint16_t>> in(6),out(3);
  for(unsigned i=0;i<6;i++){size_t elements=1;for(unsigned d=0;d<g.inputs[i].v1.rank;d++)elements*=g.inputs[i].v1.dimensions[d];in[i].assign(elements,i==5?0:32768);g.inputs[i].v1.clientBuf={in[i].data(),uint32_t(elements*(i==5?1:2))};}
  for(unsigned i=0;i<3;i++){size_t elements=1;for(unsigned d=0;d<g.outputs[i].v1.rank;d++)elements*=g.outputs[i].v1.dimensions[d];out[i].resize(elements);g.outputs[i].v1.clientBuf={out[i].data(),uint32_t(elements*2)};}
  std::vector<std::unique_ptr<SharedTensor>> shared;if(cfg.value("shared_kv",false))for(unsigned i=1;i<=2;i++)shared.push_back(share(g.context,g.inputs[i],in[i].data(),in[i].size()*2));Qnn_ProfileHandle_t profile=nullptr;check(api.profileCreate(backend,QNN_PROFILE_LEVEL_DETAILED,&profile),"profile create");auto begin=std::chrono::steady_clock::now();unsigned repeats=cfg.value("repeats",10u);for(unsigned i=0;i<repeats;i++)check(api.graphExecute(g.graph,g.inputs.data(),6,g.outputs.data(),3,profile,nullptr),"profile graph");printf("PROFILE repeats=%u mean_ms=%.3f\n",repeats,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/repeats);std::function<void(QnnProfile_EventId_t)> visit=[&](QnnProfile_EventId_t id){QnnProfile_EventData_t data=QNN_PROFILE_EVENT_DATA_INIT;check(api.profileGetEventData(id,&data),"profile event");printf("EVENT type=%u unit=%u value=%llu name=%s\n",data.type,data.unit,(unsigned long long)data.value,data.identifier?data.identifier:"");const QnnProfile_EventId_t*children=nullptr;uint32_t size=0;check(api.profileGetSubEvents(id,&children,&size),"sub events");for(unsigned j=0;j<size;j++)visit(children[j]);};const QnnProfile_EventId_t*events=nullptr;uint32_t count2=0;check(api.profileGetEvents(profile,&events,&count2),"events");for(unsigned i=0;i<count2;i++)visit(events[i]);api.profileFree(profile);shared.clear();api.contextFree(g.context,nullptr);system.systemContextFree(g.system);return 0;
 }
 if(cfg.contains("draft_probe_dir")){
  std::string directory=cfg.at("draft_probe_dir");json fixture;std::ifstream(directory+"/fixture.json")>>fixture;
  {DSparkRuntime draft(api,load,[&](CachedGraph&g){api.contextFree(g.context,nullptr);system.systemContextFree(g.system);},cfg.at("dspark_cache_dir"),capacity);
   auto features=read<float>(directory+"/features.f32"),noise=read<float>(directory+"/noise.f32"),reference=read<float>(directory+"/hidden.f32");unsigned count=fixture.at("context_rows");if(features.size()!=(size_t)count*5*hidden_size||noise.size()!=(size_t)draft.rows*hidden_size)throw std::runtime_error("草稿 oracle 输入尺寸错误");
   for(unsigned p=0;p<count;p++){std::vector<std::vector<uint16_t>>encoded(5,std::vector<uint16_t>(hidden_size));std::vector<const uint16_t*>ptrs;auto&projector=draft.graphs[p<4?p:4];for(unsigned l=0;l<5;l++){auto e=projector.inputs[l].v1.quantizeParams.scaleOffsetEncoding;for(unsigned c=0;c<hidden_size;c++)encoded[l][c]=std::clamp((int)std::round(features[((size_t)p*5+l)*hidden_size+c]/e.scale)-e.offset,0,65535);ptrs.push_back(encoded[l].data());}draft.append(ptrs,1,p);}
   std::vector<uint16_t>embedding(noise.size());auto e=draft.graphs[8].inputs[0].v1.quantizeParams.scaleOffsetEncoding;for(size_t i=0;i<noise.size();i++)embedding[i]=std::clamp((int)std::round(noise[i]/e.scale)-e.offset,0,65535);
   auto begin=std::chrono::steady_clock::now();auto proposal=draft.propose(embedding.data(),fixture.at("first_token"));double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();e=draft.graphs[13].outputs[0].v1.quantizeParams.scaleOffsetEncoding;double error=0,energy=0;for(size_t i=0;i<reference.size();i++){double d=((int)draft.normed[i]+e.offset)*e.scale-reference[i];error+=d*d;energy+=reference[i]*reference[i];}json result={{"tokens",proposal},{"reference",fixture.at("tokens")},{"hidden_relative_rmse",std::sqrt(error/std::max(energy,1e-20))},{"draft_ms",ms}};printf("DSPARK_ORACLE %s\n",result.dump().c_str());
  }return 0;
 }
 json active_request=cfg;
 std::string buffer,request_line;
 for(;;){
 cfg=active_request;
 shared_group=nullptr;max_spill=0;
 capacity=cfg.value("resident_kv_capacity",4096u);
 // 常驻模式反复复用已加载的图；每个请求必须与首次配置的图身份一致，否则报错回退一次性进程。
 json cfg_identity=json::object();for(const char*key:{"root","context_cache_dir","model_layers","resident_kv_capacity","shared_kv","shared_kv_layers","shared_kv_arena","shared_kv_flat","share_spill_fill","context_io_estimation","rope_theta","performance_vote","performance_turbo","performance_max","embedding_cache_parts"})if(cfg.contains(key))cfg_identity[key]=cfg.at(key);
 cfg_identity["embedding_cache_dir"]=cfg.value("embedding_cache_dir",cfg.at("context_cache_dir").get<std::string>());
 if(cfg.value("share_spill_fill",false)){
  std::vector<std::string>paths;for(unsigned i=0;i<layers;i++)paths.push_back(cfg.at("context_cache_dir").get<std::string>()+"/decode_layer_"+std::to_string(i)+".bin");paths.push_back(cfg.at("context_cache_dir").get<std::string>()+"/decode_head.bin");
  for(unsigned i=0;i<80;i++)paths.push_back(cfg.value("embedding_cache_dir",cfg.at("context_cache_dir").get<std::string>())+"/embedding_"+std::to_string(i)+".bin");
  for(auto&path:paths){auto bytes=read<uint8_t>(path);QnnSystemContext_Handle_t metadata=nullptr;check(system.systemContextCreate(&metadata),"spill metadata");const QnnSystemContext_BinaryInfo_t*binary=nullptr;Qnn_ContextBinarySize_t size=0;check(system.systemContextGetBinaryInfo(metadata,bytes.data(),bytes.size(),&binary,&size),"spill binary info");max_spill=std::max(max_spill,spill_size(binary));system.systemContextFree(metadata);}
  printf("SHARED_SPILL bytes=%llu\n",(unsigned long long)max_spill);fflush(stdout);
 }
 auto load_begin=std::chrono::steady_clock::now();std::vector<CachedGraph> graphs;CachedGraph head;std::map<uint32_t,CachedGraph> embeddings;std::vector<LayerBuffers> buffers(layers);
 ScopeExit cleanup{[&]{for(auto&b:buffers){b.shared_key.reset();b.shared_value.reset();}for(auto&entry:embeddings){api.contextFree(entry.second.context,nullptr);system.systemContextFree(entry.second.system);}if(head.context)api.contextFree(head.context,nullptr);if(head.system)system.systemContextFree(head.system);for(auto i=graphs.rbegin();i!=graphs.rend();++i){api.contextFree(i->context,nullptr);system.systemContextFree(i->system);}kv_arena.reset();}};
 {std::string cache_dir=cfg.at("context_cache_dir").get<std::string>();graphs.reserve(layers);for(unsigned i=0;i<layers;i++)graphs.push_back(load(cache_dir+"/decode_layer_"+std::to_string(i)+".bin",6,3));head=load(cache_dir+"/decode_head.bin",1,1);}
 std::map<uint32_t,uint64_t> embedding_used;uint64_t embedding_clock=0;unsigned embedding_limit=std::clamp(cfg.value("embedding_cache_parts",8u),1u,16u);std::string embedding_dir=cfg.value("embedding_cache_dir",cfg.at("context_cache_dir").get<std::string>());double load_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-load_begin).count();
 for(unsigned i=0;i<layers;i++){require_same_encoding(graphs[i].outputs[0],graphs[i].inputs[1],"layer="+std::to_string(i)+" key");require_same_encoding(graphs[i].outputs[1],graphs[i].inputs[2],"layer="+std::to_string(i)+" value");require_same_encoding(graphs[i].outputs[2],i+1<layers?graphs[i+1].inputs[0]:head.inputs[0],"layer="+std::to_string(i)+" hidden");}
 unsigned batch=graphs[0].inputs[0].v1.dimensions[0];if(!batch||batch>32||head.inputs[0].v1.dimensions[0]!=batch)throw std::runtime_error("模型层与输出头 batch 不匹配");bool transposed_kv=graphs[0].inputs[1].v1.rank==3;unsigned kv_slots=transposed_kv?capacity-1:capacity;
 unsigned mask_cols=graphs[0].inputs[5].v1.dimensions[1];if(mask_cols!=capacity+batch-1)throw std::runtime_error("批量 attention 容量与 KV 不匹配");
 for(unsigned i=0;i<layers;i++){auto&b=buffers[i];b.hidden_in.resize((size_t)batch*hidden_size);b.hidden_out.resize((size_t)batch*hidden_size);b.key_out.resize((size_t)batch*kv_cols);b.value_out.resize((size_t)batch*kv_cols);b.key_in.assign((size_t)kv_slots*kv_cols,32768);b.value_in.assign((size_t)kv_slots*kv_cols,32768);if(cfg.value("shared_kv",false)&&i<cfg.value("shared_kv_layers",layers)){b.shared_key=share(graphs[i].context,graphs[i].inputs[1],b.key_in.data(),b.key_in.size()*2);b.shared_value=share(graphs[i].context,graphs[i].inputs[2],b.value_in.data(),b.value_in.size()*2);b.key_in.clear();b.key_in.shrink_to_fit();b.value_in.clear();b.value_in.shrink_to_fit();}}
 std::vector<uint16_t>logits((size_t)batch*vocab),cosine((size_t)batch*head_dim),sine((size_t)batch*head_dim);std::vector<uint8_t>mask((size_t)batch*mask_cols);
 // 每请求状态；lambda 通过引用看到 serve() 换入的最新值。
 json state;unsigned position=0;uint32_t token=0;std::vector<uint32_t> generated_tokens;std::string audit_dir;std::vector<unsigned> audit_positions;
 auto fill_rope=[&](unsigned p){float theta=cfg.value("rope_theta",5000000.0f),scale=1.0f/32767;for(unsigned row=0;row<batch;row++)for(unsigned d=0;d<head_dim/2;d++){float angle=(p+row)/std::pow(theta,2.0f*d/head_dim);uint16_t c=std::clamp((int)std::round(std::cos(angle)/scale)+32768,0,65535),s=std::clamp((int)std::round(std::sin(angle)/scale)+32768,0,65535);cosine[row*head_dim+2*d]=cosine[row*head_dim+2*d+1]=c;sine[row*head_dim+2*d]=sine[row*head_dim+2*d+1]=s;}};
 auto fill_mask=[&](unsigned p){std::fill(mask.begin(),mask.end(),1);for(unsigned row=0;row<batch;row++){std::fill_n(mask.begin()+(size_t)row*mask_cols,p,0);std::fill_n(mask.begin()+(size_t)row*mask_cols+kv_slots,row+1,0);}};
 auto embed=[&](uint32_t wanted,unsigned destination_row=0){if(wanted>=vocab)throw std::runtime_error("embedding token 越界: "+std::to_string(wanted));auto&dst=buffers[0].hidden_in;uint32_t global_group=wanted/32,part=global_group/51,group=global_group%51,row=wanted%32;
  // 限制常驻 context 数量，避免整张词表的 80 个分片耗尽 HTP 资源。
  if(!embeddings.count(part)){
   if(embeddings.size()>=embedding_limit){auto oldest=std::min_element(embedding_used.begin(),embedding_used.end(),[](auto&a,auto&b){return a.second<b.second;});auto&evicted=embeddings.at(oldest->first);check(api.contextFree(evicted.context,nullptr),"free embedding");check(system.systemContextFree(evicted.system),"free embedding metadata");embeddings.erase(oldest->first);embedding_used.erase(oldest->first);}
   embeddings.emplace(part,load(embedding_dir+"/embedding_"+std::to_string(part)+".bin",2,1));
  }
  embedding_used[part]=++embedding_clock;auto&embedding=embeddings.at(part);require_same_encoding(embedding.outputs[0],graphs[0].inputs[0],"embedding");
  embedding.inputs[0].v1.clientBuf={&group,4};embedding.inputs[1].v1.clientBuf={&row,4};embedding.outputs[0].v1.clientBuf={dst.data()+(size_t)destination_row*hidden_size,uint32_t(hidden_size*2)};check(api.graphExecute(embedding.graph,embedding.inputs.data(),2,embedding.outputs.data(),1,nullptr,nullptr),("embedding token="+std::to_string(wanted)+" part="+std::to_string(part)).c_str());};
 auto run=[&](unsigned active=1,bool output=true){for(unsigned i=0;i<layers;i++){auto&g=graphs[i];auto&b=buffers[i];void*in_ptrs[]={b.hidden_in.data(),b.key(),b.value(),cosine.data(),sine.data(),mask.data()};uint32_t in_sizes[]={uint32_t(b.hidden_in.size()*2),uint32_t((size_t)kv_slots*kv_cols*2),uint32_t((size_t)kv_slots*kv_cols*2),uint32_t(cosine.size()*2),uint32_t(sine.size()*2),uint32_t(mask.size())};void*out_ptrs[]={b.key_out.data(),b.value_out.data(),b.hidden_out.data()};uint32_t out_sizes[]={uint32_t(b.key_out.size()*2),uint32_t(b.value_out.size()*2),uint32_t(b.hidden_out.size()*2)};for(unsigned j=0;j<6;j++)if(g.inputs[j].v1.memType!=QNN_TENSORMEMTYPE_MEMHANDLE)g.inputs[j].v1.clientBuf={in_ptrs[j],in_sizes[j]};for(unsigned j=0;j<3;j++)g.outputs[j].v1.clientBuf={out_ptrs[j],out_sizes[j]};check(api.graphExecute(g.graph,g.inputs.data(),g.inputs.size(),g.outputs.data(),g.outputs.size(),nullptr,nullptr),"decode layer execute");if(!audit_dir.empty()&&std::find(audit_positions.begin(),audit_positions.end(),position)!=audit_positions.end()){std::string path=audit_dir+"/p"+std::to_string(position)+"_l"+std::to_string(i);write(path+"_h.u16",b.hidden_out);write(path+"_k.u16",b.key_out);write(path+"_v.u16",b.value_out);json scales={{"hidden",g.outputs[2].v1.quantizeParams.scaleOffsetEncoding.scale},{"key",g.outputs[0].v1.quantizeParams.scaleOffsetEncoding.scale},{"value",g.outputs[1].v1.quantizeParams.scaleOffsetEncoding.scale}};std::ofstream(path+".json")<<scales.dump();}if(i+1<layers)buffers[i+1].hidden_in=b.hidden_out;}if(!output)return;head.inputs[0].v1.clientBuf={buffers.back().hidden_out.data(),uint32_t((size_t)batch*hidden_size*2)};head.outputs[0].v1.clientBuf={logits.data(),uint32_t(logits.size()*2)};check(api.graphExecute(head.graph,head.inputs.data(),1,head.outputs.data(),1,nullptr,nullptr),"decode head execute");auto first=logits.begin()+(size_t)(active-1)*vocab;token=std::max_element(first,first+vocab)-first;};
 std::unique_ptr<DSparkRuntime> draft;
 // 两个槽交替写入，最后原子替换元数据；中途断电仍可恢复上一份完整 KV。
 auto checkpoint=[&]{
  std::string destination=cfg.value("state_out_dir",std::string());if(destination.empty())return;
  if(mkdir(destination.c_str(),0700)&&errno!=EEXIST)throw std::runtime_error("无法创建 KV 保存目录");
  std::string previous=state.value("kv_dir",std::string()),slot=destination+(previous==destination+"/a"?"/b":"/a");
  if(mkdir(slot.c_str(),0700)&&errno!=EEXIST)throw std::runtime_error("无法创建 KV 保存槽");
  auto durable_write=[&](const std::string&path,const void*data,size_t size){int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_TRUNC,0600);if(fd<0)throw std::runtime_error("无法保存 KV: "+path);auto*bytes=static_cast<const char*>(data);while(size){ssize_t n=::write(fd,bytes,size);if(n<0&&errno==EINTR)continue;if(n<=0){close(fd);throw std::runtime_error("KV 写入失败: "+path);}bytes+=n;size-=n;}int rc=fsync(fd);close(fd);if(rc)throw std::runtime_error("KV fsync 失败: "+path);};
  // 磁盘保持逐 token 的规范布局，设备端排布变化不改变保存格式。
  std::vector<uint16_t> canonical_key(transposed_kv?(size_t)position*kv_cols:0),canonical_value(canonical_key.size());
  for(unsigned i=0;i<layers;i++){auto&b=buffers[i];if(transposed_kv)for(unsigned r=0;r<position;r++)for(unsigned c=0;c<kv_cols;c++){canonical_key[(size_t)r*kv_cols+c]=b.key()[(size_t)c*kv_slots+r];canonical_value[(size_t)r*kv_cols+c]=b.value()[((size_t)(c/head_dim)*kv_slots+r)*head_dim+c%head_dim];}durable_write(slot+"/k"+std::to_string(i)+".u16",transposed_kv?canonical_key.data():b.key(),(size_t)position*kv_cols*2);durable_write(slot+"/v"+std::to_string(i)+".u16",transposed_kv?canonical_value.data():b.value(),(size_t)position*kv_cols*2);state["kv"][std::to_string(i)]["rows"]=position;}
  if(draft){if(draft->position!=position)throw std::runtime_error("目标与草稿 KV checkpoint 位置不一致");draft->checkpoint(slot,durable_write);}
  int dir_fd=open(slot.c_str(),O_RDONLY|O_DIRECTORY);if(dir_fd>=0){fsync(dir_fd);close(dir_fd);}
  state["schema_version"]=1;state["kv_dir"]=slot;state["prompt_tokens"]=position;state["first_token"]=token;state["generated_tokens"]=generated_tokens;state["context_capacity"]=capacity;state["cache_identity"]=cfg.value("cache_identity",std::string());state["request_id"]=cfg.value("request_id",std::string());
  std::string encoded=state.dump(),path=destination+"/state.json";durable_write(path+".tmp",encoded.data(),encoded.size());if(std::rename((path+".tmp").c_str(),path.c_str()))throw std::runtime_error("KV 元数据发布失败");dir_fd=open(destination.c_str(),O_RDONLY|O_DIRECTORY);if(dir_fd>=0){fsync(dir_fd);close(dir_fd);}checkpoint_requested=0;printf("CHECKPOINT position=%u\n",position);fflush(stdout);
 };
 auto append_kv=[&](unsigned active=1){for(auto&b:buffers){if(transposed_kv){for(unsigned row=0;row<active;row++)for(unsigned c=0;c<kv_cols;c++){b.key()[(size_t)c*kv_slots+position+row]=b.key_out[(size_t)row*kv_cols+c];b.value()[((size_t)(c/head_dim)*kv_slots+position+row)*head_dim+c%head_dim]=b.value_out[(size_t)row*kv_cols+c];}}else{std::copy_n(b.key_out.begin(),(size_t)active*kv_cols,b.key()+(size_t)position*kv_cols);std::copy_n(b.value_out.begin(),(size_t)active*kv_cols,b.value()+(size_t)position*kv_cols);}}if(draft){std::vector<const uint16_t*>features;for(unsigned layer:draft->target_layers)features.push_back(buffers.at(layer).hidden_out.data());draft->append(features,active,position);}};
 // 上下文阶梯:接近容量时 checkpoint,换 ladder 下一档图集重载并从 checkpoint
 // 恢复继续生成。档位图集必须与当前档共享同一份 KV 量化校准(恢复校验会把关)。
 json upgrade_resume;
 auto mem_available_mb=[&]{std::ifstream meminfo("/proc/meminfo");std::string row;while(std::getline(meminfo,row))if(row.rfind("MemAvailable:",0)==0){unsigned long kb=0;try{kb=std::stoul(row.substr(13));}catch(...){}return kb/1024;}return 0ul;};
 auto next_rung=[&](const json&request)->json{
  if(!request.contains("context_ladder"))return json();
  json pick;unsigned best=0;
  for(const json&rung:request.at("context_ladder")){
   unsigned rung_capacity=rung.value("capacity",0u);
   if(rung_capacity<=capacity)continue;
   unsigned need=rung.value("min_mem_mb",0u);
   if(need&&(unsigned)mem_available_mb()<need)continue;
   if(!best||rung_capacity<best){best=rung_capacity;pick=rung;}
  }
  return pick;
 };
 auto request_upgrade=[&](const json&request,const std::vector<uint32_t>&pending_suffix)->bool{
  if(request.value("state_out_dir",std::string()).empty())return false;
  json rung=next_rung(request);if(rung.is_null())return false;
  checkpoint();
  upgrade_resume=request;
  upgrade_resume["context_cache_dir"]=rung.at("context_cache_dir");
  upgrade_resume["resident_kv_capacity"]=rung.at("capacity");
  if(rung.contains("embedding_cache_dir"))upgrade_resume["embedding_cache_dir"]=rung.at("embedding_cache_dir");
  // 大容量档的 KV 内存形态可以不同(8K 档 rpcmem 注册超限,需关闭共享 KV)。
  for(const char*key:{"shared_kv","shared_kv_arena","shared_kv_flat","shared_kv_layers"})if(rung.contains(key))upgrade_resume[key]=rung.at(key);
  upgrade_resume["state_dir"]=request.at("state_out_dir");
  upgrade_resume["prompt_suffix_tokens"]=pending_suffix;
  upgrade_resume["suppress_first_token"]=true;
  upgrade_resume.erase("prompt_tokens");upgrade_resume.erase("prefix_state_dir");upgrade_resume.erase("dspark_cache_dir");
  printf("CONTEXT_UPGRADE from_capacity=%u to_capacity=%u position=%u\n",capacity,rung.at("capacity").get<unsigned>(),position);fflush(stdout);
  return true;
 };
 printf("LOAD contexts=%u batch=%u wall_ms=%.3f%s\n",layers+1,batch,load_ms,server?" resident=1":"");fflush(stdout);
 // 单个请求的完整执行；常驻模式反复调用，图与共享 KV 注册只在进程内完成一次。
 auto serve=[&](const json&request)->int{
  // 常驻会话只能复用与首次加载完全相同的图与内存布局；不一致必须回退一次性进程。
  if(server){json incoming=json::object();for(auto&item:cfg_identity.items()){json fallback=item.key()=="embedding_cache_dir"?json(request.value("context_cache_dir",std::string())):json();incoming[item.key()]=request.contains(item.key())?request.at(item.key()):fallback;}if(incoming!=cfg_identity)throw std::runtime_error("请求配置与常驻图不一致，需回退一次性执行");}
  cfg=request;
  bool fresh=request.contains("prompt_tokens"),restore_prefix=fresh&&request.contains("prefix_state_dir");std::vector<uint32_t> prefix_tokens;
  if(restore_prefix){prefix_tokens=request.at("prefix_tokens").get<std::vector<uint32_t>>();auto supplied=request.at("prompt_tokens").get<std::vector<uint32_t>>();if(prefix_tokens.empty()||supplied.size()<=prefix_tokens.size()||!std::equal(prefix_tokens.begin(),prefix_tokens.end(),supplied.begin()))throw std::runtime_error("缓存前缀与当前 prompt 不一致");fresh=false;}
  if(!fresh){std::string initial=restore_prefix?request.at("prefix_state_dir").get<std::string>():request.at("state_dir").get<std::string>();unsigned wait_ms=request.value("state_wait_ms",0u),waited=0;while(!std::ifstream(initial+"/state.json")&&waited<wait_ms){usleep(10000);waited+=10;}std::ifstream sf(initial+"/state.json");if(!sf)throw std::runtime_error("缺少 prefill state");sf>>state;
   if(restore_prefix&&(!state.contains("cache_identity")||state.at("prompt_tokens").get<unsigned>()!=prefix_tokens.size()))throw std::runtime_error("固定前缀缓存元数据不完整");if(state.contains("cache_identity")&&state.at("cache_identity")!=(restore_prefix?request.at("prefix_cache_identity").get<std::string>():request.value("cache_identity",std::string())))throw std::runtime_error("保存的 KV 与当前模型或 NPU 图不兼容");}
  audit_dir=request.value("audit_dir",std::string());audit_positions=request.value("audit_positions",std::vector<unsigned>{});if(!audit_dir.empty())mkdir(audit_dir.c_str(),0755);
  stop_tokens=request.value("stop_tokens",std::vector<uint32_t>{1,130073});
  if(fresh){state={{"prompt_tokens",0},{"first_token",0},{"layers",layers}};for(unsigned i=0;i<layers;i++)state["kv"][std::to_string(i)]={{"rows",0},{"cols",kv_cols},{"key_step",graphs[i].inputs[1].v1.quantizeParams.scaleOffsetEncoding.scale},{"value_step",graphs[i].inputs[2].v1.quantizeParams.scaleOffsetEncoding.scale}};}
  std::string kv_dir=state.value("kv_dir",request.value("state_dir",std::string()));position=state.at("prompt_tokens").get<unsigned>();token=state.at("first_token");
  if((!fresh&&position==0)||position>=capacity)throw std::runtime_error("prefill 已占满 KV capacity="+std::to_string(capacity));
  for(unsigned i=0;i<layers;i++){auto&g=graphs[i];auto&km=state.at("kv").at(std::to_string(i));if((transposed_kv?(g.inputs[1].v1.rank!=3||g.inputs[1].v1.dimensions[2]!=kv_slots||g.inputs[1].v1.dimensions[0]*g.inputs[1].v1.dimensions[1]!=kv_cols):g.inputs[1].v1.dimensions[0]!=capacity)||g.inputs[1].v1.quantizeParams.scaleOffsetEncoding.scale!=km.at("key_step").get<float>()||g.inputs[2].v1.quantizeParams.scaleOffsetEncoding.scale!=km.at("value_step").get<float>())throw std::runtime_error("prefill/decode KV 不匹配 layer="+std::to_string(i)+" rows="+std::to_string(g.inputs[1].v1.dimensions[0])+" key="+std::to_string(g.inputs[1].v1.quantizeParams.scaleOffsetEncoding.scale)+" expected="+std::to_string(km.at("key_step").get<float>())+" value="+std::to_string(g.inputs[2].v1.quantizeParams.scaleOffsetEncoding.scale)+" expected="+std::to_string(km.at("value_step").get<float>()));}
  for(unsigned i=0;i<layers;i++){auto&b=buffers[i];auto k=fresh?std::vector<uint16_t>{}:read<uint16_t>(kv_dir+"/k"+std::to_string(i)+".u16"),v=fresh?std::vector<uint16_t>{}:read<uint16_t>(kv_dir+"/v"+std::to_string(i)+".u16");if(k.size()!=v.size()||k.size()>(size_t)capacity*kv_cols||k.size()<(size_t)position*kv_cols||k.size()%kv_cols)throw std::runtime_error("KV 文件长度无效 layer="+std::to_string(i));
   // 每请求先把整块 KV 恢复到中性值，等价于一次性进程里新分配的 buffer。
   if(b.shared_key){std::fill_n(b.key(),(size_t)kv_slots*kv_cols,uint16_t(32768));std::fill_n(b.value(),(size_t)kv_slots*kv_cols,uint16_t(32768));}else{b.key_in.assign((size_t)kv_slots*kv_cols,uint16_t(32768));b.value_in.assign((size_t)kv_slots*kv_cols,uint16_t(32768));}
   if(transposed_kv){for(unsigned r=0;r<position;r++)for(unsigned c=0;c<kv_cols;c++){b.key()[(size_t)c*kv_slots+r]=k[(size_t)r*kv_cols+c];b.value()[((size_t)(c/head_dim)*kv_slots+r)*head_dim+c%head_dim]=v[(size_t)r*kv_cols+c];}}else{std::copy_n(k.begin(),(size_t)position*kv_cols,b.key());std::copy_n(v.begin(),(size_t)position*kv_cols,b.value());}}
  draft.reset();std::string dspark_dir=request.value("dspark_cache_dir",std::string());
  if(!dspark_dir.empty()){
   if(server)throw std::runtime_error("常驻模式暂不支持 DSpark 请求，请回退一次性执行");
   if((batch!=4&&batch!=8)||!transposed_kv)throw std::runtime_error("DSpark 需要匹配的 AR4/AR8 目标验证图");
   draft=std::make_unique<DSparkRuntime>(api,load,[&](CachedGraph&g){api.contextFree(g.context,nullptr);system.systemContextFree(g.system);},dspark_dir,capacity);
   draft->markov_enabled=!request.value("dspark_disable_markov",false);
   for(unsigned i=0;i<draft->target_layers.size();i++)require_same_encoding(graphs.at(draft->target_layers[i]).outputs[2],draft->graphs[5].inputs[i],"DSpark 目标 hidden");
   require_same_encoding(graphs[0].inputs[0],draft->graphs[8].inputs[0],"DSpark embedding");
   if(position==4&&restore_prefix){draft->restore_prefix();draft->discard_prefix_graphs();}else if(position){draft->restore(kv_dir,position);draft->discard_prefix_graphs();}
   printf("MTP_ENABLED algorithm=DSpark draft_layers=%u draft_tokens=%u verify_rows=%u markov=%d\n",draft->layers,draft->rows,batch,draft->markov_enabled?1:0);fflush(stdout);
  }
  // 长输入先消费剩余提示词；中间的预测不能作为回答发给用户。
  auto suffix=request.value(fresh?"prompt_tokens":"prompt_suffix_tokens",std::vector<uint32_t>{});if(restore_prefix){suffix=request.at("prompt_tokens").get<std::vector<uint32_t>>();suffix.erase(suffix.begin(),suffix.begin()+prefix_tokens.size());printf("PREFIX_RESTORE tokens=%zu\n",prefix_tokens.size());}if(fresh&&suffix.empty())throw std::runtime_error("prompt 不能为空");
  if(suffix.size()>=capacity-position){if(request_upgrade(request,suffix))return -2;throw std::runtime_error("输入未给回答预留上下文空间");}
  auto append_begin=std::chrono::steady_clock::now();
  for(size_t i=0;i<suffix.size();){unsigned active=std::min<size_t>(batch,suffix.size()-i);if(i&&i%128==0){printf("PREFILL_PROGRESS consumed=%zu total=%zu\n",i,suffix.size());fflush(stdout);}if(cancel_requested)throw std::runtime_error("prefill 已取消，保留上一轮 KV");for(unsigned row=0;row<active;row++)embed(suffix[i+row],row);fill_rope(position);fill_mask(position);run(active,i+active==suffix.size());append_kv(active);position+=active;i+=active;}
  if(!suffix.empty())printf("APPEND_PREFILL tokens=%zu wall_ms=%.3f\n",suffix.size(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-append_begin).count());
  generated_tokens.assign(1,token);
  // 升级恢复时首个 token 在换档前已输出过,重放会令流式显示重复一段文本。
  if(!request.value("suppress_first_token",false))printf("TOKEN position=%u actual=%u\n",position,token);fflush(stdout);auto decode_begin=std::chrono::steady_clock::now();unsigned generated_steps=0;
  auto stopped=[&]{return std::find(stop_tokens.begin(),stop_tokens.end(),token)!=stop_tokens.end();};
  // prefill 已产生一个回答 token，它也占用总上下文预算。
  unsigned steps=request.value("resident_decode_steps",8u),available=capacity-position-1,limit=steps?std::min(steps,available):available;
  unsigned max_new_tokens=request.value("max_new_tokens",0u);if(max_new_tokens)limit=std::min(limit,max_new_tokens-1);
  for(;generated_steps<limit&&!stopped()&&!cancel_requested;){
   if(capacity-position<=request.value("context_upgrade_margin",256u)&&request_upgrade(request,{}))return -2;
   if(draft){
    auto started=std::chrono::steady_clock::now();std::vector<uint16_t>draft_embedding((size_t)draft->rows*hidden_size);embed(token,0);std::copy_n(buffers[0].hidden_in.data(),hidden_size,draft_embedding.data());embed(draft->spec.at("mask_token_id"),0);for(unsigned r=1;r<draft->rows;r++)std::copy_n(buffers[0].hidden_in.data(),hidden_size,draft_embedding.data()+(size_t)r*hidden_size);auto proposal=draft->propose(draft_embedding.data(),token,batch-1);double draft_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    embed(token,0);for(unsigned r=0;r<proposal.size();r++)embed(proposal[r],r+1);fill_rope(position);fill_mask(position);auto verify_begin=std::chrono::steady_clock::now();run(batch);double verify_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-verify_begin).count();
    auto predicted=[&](unsigned row){auto first=logits.begin()+(size_t)row*vocab;return uint32_t(std::max_element(first,first+vocab)-first);};unsigned accepted=0;while(accepted<proposal.size()&&proposal[accepted]==predicted(accepted))accepted++;
    std::vector<uint32_t>emitted(proposal.begin(),proposal.begin()+accepted);emitted.push_back(predicted(accepted));unsigned remaining=limit-generated_steps;if(emitted.size()>remaining)emitted.resize(remaining);for(unsigned i=0;i<emitted.size();i++)if(std::find(stop_tokens.begin(),stop_tokens.end(),emitted[i])!=stop_tokens.end()){emitted.resize(i+1);break;}
    unsigned committed=emitted.size(),before=generated_steps;append_kv(committed);for(auto next_token:emitted){position++;generated_steps++;token=next_token;generated_tokens.push_back(token);printf("TOKEN position=%u actual=%u\n",position,token);}fflush(stdout);printf("MTP_ROUND proposed=%zu accepted=%u emitted=%u draft_ms=%.3f verify_ms=%.3f\n",proposal.size(),accepted,committed,draft_ms,verify_ms);fflush(stdout);
    if(request.value("dspark_diagnose",false)){std::string proposals,predictions,confidences;for(auto value:proposal)proposals+=std::to_string(value)+",";for(unsigned r=0;r<batch;r++)predictions+=std::to_string(predicted(r))+",";for(auto value:draft->confidences)confidences+=std::to_string(value)+",";printf("MTP_DIAG position=%u proposals=%s predicted=%s confidence=%s\n",position,proposals.c_str(),predictions.c_str(),confidences.c_str());fflush(stdout);}
    if(checkpoint_requested||before/128!=generated_steps/128)checkpoint();continue;
   }
   embed(token);fill_rope(position);fill_mask(position);run();append_kv();position++;generated_steps++;generated_tokens.push_back(token);printf("TOKEN position=%u actual=%u\n",position,token);fflush(stdout);
   if(checkpoint_requested||generated_steps%128==0)checkpoint();
  }
  double decode_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-decode_begin).count();checkpoint();printf("DECODE tokens=%u wall_ms=%.3f tok_s=%.3f\n",generated_steps,decode_ms,generated_steps?1000.0*generated_steps/decode_ms:0.0);printf("FINISH reason=%s\n",cancel_requested?"cancelled":(stopped()?"stop":(limit==available?"context_length":"length")));
  return 0;
 };
 int rc=serve(active_request);
 if(rc==-2){active_request=upgrade_resume;continue;}
 if(!server)return rc;
 printf("REQUEST_DONE\n");fflush(stdout);
 if(cancel_requested)return 0;
 bool reload=false;
 while(!reload){
  int got=htp_read_line_poll(STDIN_FILENO,buffer,request_line,200);
  if(got<0)break;
  if(got==0){if(cancel_requested)break;continue;}
  if(request_line.empty()||request_line=="QUIT")break;
  try{std::ifstream rf(request_line);if(!rf)throw std::runtime_error("无法读取请求配置 "+request_line);json request;rf>>request;if(serve(request)==-2)reload=true;}
  catch(std::exception&failure){fprintf(stderr,"ERROR: %s\n",failure.what());fflush(stderr);}
  if(reload)break;
  printf("REQUEST_DONE\n");fflush(stdout);
  if(cancel_requested)break;
 }
 if(reload){active_request=upgrade_resume;continue;}
 return 0;
 }
}catch(std::exception&e){fprintf(stderr,"ERROR: %s\n",e.what());return 2;}}
