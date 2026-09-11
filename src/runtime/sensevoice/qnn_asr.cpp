// SenseVoice 常驻 ASR runner(原 zllm-bench qnn_htp_cached_asr):加载离线
// 分段 context binary 链,消费 fbank 特征文件、写回逐帧 token id。CTC 合并与
// 文本拼接由 host(zllm android C ABI)完成。由 zllm-qnn-asr bin 经 C ABI 调用。
#include "htp_graph.hpp"
#include "htp_session.hpp"
#include <algorithm>
#include <csignal>
#include <unistd.h>

struct Segment {
 QnnSystemContext_Handle_t metadata=nullptr;
 Qnn_ContextHandle_t context=nullptr;
 Qnn_GraphHandle_t graph=nullptr;
 Qnn_Tensor_t input=QNN_TENSOR_INIT,output=QNN_TENSOR_INIT;
 std::vector<uint16_t> values;
};

static size_t elements(const Qnn_Tensor_t&t){
 if(t.version!=QNN_TENSOR_VERSION_1||t.v1.dataType!=QNN_DATATYPE_UFIXED_POINT_16||t.v1.quantizeParams.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET||!(t.v1.quantizeParams.scaleOffsetEncoding.scale>0))throw std::runtime_error("ASR 缓存必须使用单一 scale 的 U16 tensor");
 size_t n=1;for(unsigned d=0;d<t.v1.rank;d++)n*=t.v1.dimensions[d];return n;
}

extern "C" int zllm_qnn_asr_main(int argc,char**argv){
 std::unique_ptr<HtpSession> session;
 std::vector<Segment> segments;
 auto cleanup=[&](){if(!session)return;auto&api=session->api;auto&system=session->system;for(auto&g:segments){if(g.context)api.contextFree(g.context,nullptr);if(g.metadata)system.systemContextFree(g.metadata);}segments.clear();};
 bool server=argc==3&&std::string(argv[1])=="--server";
 try{
  if(!((argc==3&&std::string(argv[1])=="--config")||server))throw std::runtime_error("需要 --config <json> 或 --server <json>");
  std::signal(SIGTERM,[](int){_exit(0);});
  std::ifstream f(argv[2]);json cfg;f>>cfg;
  auto paths=cfg.at("graphs").get<std::vector<std::string>>();
  auto inputs=cfg.at("input_files").get<std::vector<std::string>>();
  if(paths.empty()||inputs.empty())throw std::runtime_error("缺少 ASR 图或音频特征");
  auto begin=std::chrono::steady_clock::now();
  session=std::make_unique<HtpSession>(htp_session_open(cfg,true));
  auto&api=session->api;auto&system=session->system;Qnn_BackendHandle_t backend=session->backend;Qnn_DeviceHandle_t device=session->device;
  segments.reserve(paths.size());
  for(auto&path:paths){
   segments.emplace_back();auto&g=segments.back();auto bytes=read<uint8_t>(path);
   check(system.systemContextCreate(&g.metadata),"metadata");const QnnSystemContext_BinaryInfo_t*binary=nullptr;Qnn_ContextBinarySize_t size=0;
   check(system.systemContextGetBinaryInfo(g.metadata,bytes.data(),bytes.size(),&binary,&size),"binary info");
   QnnSystemContext_GraphInfo_t*info=nullptr;unsigned graphs=0;
   switch(binary->version){
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:info=binary->contextBinaryInfoV1.graphs;graphs=binary->contextBinaryInfoV1.numGraphs;break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:info=binary->contextBinaryInfoV2.graphs;graphs=binary->contextBinaryInfoV2.numGraphs;break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:info=binary->contextBinaryInfoV3.graphs;graphs=binary->contextBinaryInfoV3.numGraphs;break;
    default:throw std::runtime_error("ASR binary 版本不支持");
   }
   if(graphs!=1)throw std::runtime_error("ASR segment 必须包含一张图: "+path);
   const char*name=nullptr;
   auto set=[&](auto&v){if(v.numGraphInputs!=1||v.numGraphOutputs!=1)throw std::runtime_error("ASR segment 必须单输入单输出，权重需要 STATIC: "+path+" inputs="+std::to_string(v.numGraphInputs));g.input=v.graphInputs[0];g.output=v.graphOutputs[0];name=v.graphName;};
   switch(info->version){case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:set(info->graphInfoV1);break;case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:set(info->graphInfoV2);break;case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:set(info->graphInfoV3);break;default:throw std::runtime_error("ASR graph info 版本不支持");}
   elements(g.input);g.values.resize(elements(g.output));g.output.v1.memType=QNN_TENSORMEMTYPE_RAW;g.output.v1.clientBuf={g.values.data(),uint32_t(g.values.size()*2)};
   check(api.contextCreateFromBinary(backend,device,nullptr,bytes.data(),bytes.size(),&g.context,nullptr),"load ASR context");
   check(api.graphRetrieve(g.context,name,&g.graph),"ASR graph");
  }
  for(size_t i=1;i<segments.size();i++){
   auto&a=segments[i-1].output;auto&b=segments[i].input;
   if(elements(a)!=elements(b)||a.v1.quantizeParams.scaleOffsetEncoding.scale!=b.v1.quantizeParams.scaleOffsetEncoding.scale||a.v1.quantizeParams.scaleOffsetEncoding.offset!=b.v1.quantizeParams.scaleOffsetEncoding.offset)throw std::runtime_error("ASR 跨段量化编码不一致: "+paths[i]);
  }
  double load_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  auto&last=segments.back();if(last.output.v1.rank!=2)throw std::runtime_error("ASR logits 需要二维矩阵");
  unsigned rows=last.output.v1.dimensions[0],vocab=last.output.v1.dimensions[1];
  printf("ASR_LOAD graphs=%zu rows=%u load_ms=%.3f%s\n",segments.size(),rows,load_ms,server?" resident=1":"");fflush(stdout);
  // 单个请求：只消费特征文件并写回 frame ids；ASR 图全程常驻。
  auto serve=[&](const json&request){
   auto files=request.at("input_files").get<std::vector<std::string>>();
   if(files.empty())throw std::runtime_error("缺少音频特征");
   if(server&&request.value("graphs",std::vector<std::string>{})!=paths)throw std::runtime_error("请求的 ASR 图与常驻图不一致，需回退一次性执行");
   json all=json::array();double infer_ms=0;
   for(auto&path:files){
    auto features=read<float>(path);if(features.size()!=elements(segments.front().input))throw std::runtime_error("ASR 特征尺寸不匹配: "+path);
    std::vector<uint16_t> quantized(features.size());auto q=segments.front().input.v1.quantizeParams.scaleOffsetEncoding;size_t clipped=0;
    for(size_t i=0;i<features.size();i++){if(!std::isfinite(features[i]))throw std::runtime_error("ASR 特征存在非有限值");double x=std::round(features[i]/q.scale)-q.offset;if(x<0||x>65535)clipped++;quantized[i]=std::clamp(x,0.0,65535.0);}
    auto start=std::chrono::steady_clock::now();
    for(size_t i=0;i<segments.size();i++){auto&g=segments[i];auto&source=i?segments[i-1].values:quantized;g.input.v1.memType=QNN_TENSORMEMTYPE_RAW;g.input.v1.clientBuf={source.data(),uint32_t(source.size()*2)};check(api.graphExecute(g.graph,&g.input,1,&g.output,1,nullptr,nullptr),paths[i].c_str());}
    double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();infer_ms+=elapsed;
    std::vector<uint32_t> ids(rows);for(unsigned r=0;r<rows;r++){auto first=last.values.begin()+(size_t)r*vocab;ids[r]=std::max_element(first,first+vocab)-first;}all.push_back(ids);
    printf("ASR_CHUNK rows=%u infer_ms=%.3f clipped=%zu\n",rows,elapsed,clipped);fflush(stdout);
   }
   std::string result=all.dump();std::ofstream out(request.at("output_ids").get<std::string>());out<<result;out.close();if(!out)throw std::runtime_error("写入 ASR frame ids 失败");
   printf("ASR_DONE graphs=%zu chunks=%zu load_ms=0.000 infer_ms=%.3f\n",segments.size(),files.size(),infer_ms);fflush(stdout);
  };
  serve(cfg);
  if(!server){cleanup();return 0;}
  printf("REQUEST_DONE\n");fflush(stdout);
  std::string buffer,line;
  while(true){
   int got=htp_read_line_poll(STDIN_FILENO,buffer,line,200);
   if(got<0)break;
   if(got==0)continue;
   if(line.empty()||line=="QUIT")break;
   try{std::ifstream rf(line);if(!rf)throw std::runtime_error("无法读取请求配置 "+line);json request;rf>>request;serve(request);}
   catch(std::exception&failure){fprintf(stderr,"ERROR: %s\n",failure.what());fflush(stderr);}
   printf("REQUEST_DONE\n");fflush(stdout);
  }
  cleanup();return 0;
 }catch(const std::exception&e){fprintf(stderr,"ERROR: %s\n",e.what());cleanup();return 2;}
}
