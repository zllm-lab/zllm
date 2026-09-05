#include <QnnInterface.h>
#include <QnnOpDef.h>
#include <QnnTypes.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>
#include <new>
#include <mutex>
#include <string>
#include <vector>

namespace {
using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);

struct Error { char message[512]; };

std::recursive_mutex shared_mutex;
void* shared_library = nullptr;
QNN_INTERFACE_VER_TYPE shared_api{};
Qnn_BackendHandle_t shared_backend = nullptr;
Qnn_DeviceHandle_t shared_device = nullptr;
std::array<Qnn_ContextHandle_t, 2> shared_contexts{};
uint32_t shared_users = 0;
std::atomic<uint64_t> next_graph_id{0};

void log_callback(const char* fmt, QnnLog_Level_t, uint64_t, va_list args) {
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
}

void fail(Error* error, const char* stage, uint64_t status = 0) {
    if (error != nullptr) std::snprintf(error->message, sizeof(error->message), "%s (QNN status=%llu)", stage, static_cast<unsigned long long>(status));
}

// 量化 MatMul 图(1~3 个共享输入的投影)。权重 STATIC INT8/INT4 per-output-channel,
// 输入 S8(客户端逐行动态量化,读回按 实际scale/烘焙scale 比值精确校正),输出 S8
// 用一次性校准的固定 scale(留 2.5× 余量)。S32/F16/F32 输出在本栈(SM8635+QAIRT2.38)
// 均 addNode 6005 拒绝或静默输出零,已证伪。
struct Linear {
    void* library = nullptr;
    QNN_INTERFACE_VER_TYPE api{};
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    Qnn_Tensor_t input = QNN_TENSOR_INIT;
    Qnn_Tensor_t output = QNN_TENSOR_INIT;
    Qnn_Tensor_t output2 = QNN_TENSOR_INIT;
    Qnn_Tensor_t output3 = QNN_TENSOR_INIT;
    std::array<uint32_t, 2> input_dims{};
    std::array<uint32_t, 2> weight_dims{};
    std::array<uint32_t, 2> output_dims{};
    std::array<uint32_t, 2> weight2_dims{};
    std::array<uint32_t, 2> output2_dims{};
    std::array<uint32_t, 2> weight3_dims{};
    std::array<uint32_t, 2> output3_dims{};
    std::vector<int8_t> weights;
    std::vector<float> scales;
    std::vector<Qnn_ScaleOffset_t> scale_offsets;
    std::vector<int8_t> weights2;
    std::vector<float> scales2;
    std::vector<Qnn_ScaleOffset_t> scale_offsets2;
    std::vector<int8_t> weights3;
    std::vector<float> scales3;
    std::vector<Qnn_ScaleOffset_t> scale_offsets3;
    bool dual = false;
    int32_t int32_scratch = 0;
    std::array<uint32_t, 2> axes_dims = {1, 1};
    uint64_t context_bytes = 0;
    std::string graph_name, input_name, weight_name, output_name;
    std::string weight2_name, output2_name, op_name, op2_name;
    std::string weight3_name, output3_name, op3_name;

    ~Linear() {
        std::lock_guard<std::recursive_mutex> lock(shared_mutex);
        // 图句柄由 context 持有(interface 无 graphDestroy);最后一个实例释放时整体回收。
        if (shared_users == 0 || --shared_users != 0) return;
        if (shared_api.contextFree != nullptr) for (auto context : shared_contexts) if (context != nullptr) shared_api.contextFree(context, nullptr);
        if (shared_device != nullptr && shared_api.deviceFree != nullptr) shared_api.deviceFree(shared_device);
        if (shared_backend != nullptr && shared_api.backendFree != nullptr) shared_api.backendFree(shared_backend);
        if (shared_library != nullptr) dlclose(shared_library);
        shared_library = nullptr; shared_backend = nullptr; shared_device = nullptr; shared_contexts = {};
    }
};

Qnn_Tensor_t quantized_tensor(const char* name, Qnn_TensorType_t type, Qnn_DataType_t dtype,
                              Qnn_QuantizeParams_t quant, uint32_t* dims, void* data, uint32_t bytes) {
    Qnn_Tensor_t value = QNN_TENSOR_INIT;
    value.version = QNN_TENSOR_VERSION_1;
    value.v1.name = name;
    value.v1.type = type;
    value.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    value.v1.dataType = dtype;
    value.v1.quantizeParams = quant;
    value.v1.rank = 2;
    value.v1.dimensions = dims;
    value.v1.memType = QNN_TENSORMEMTYPE_RAW;
    value.v1.clientBuf.data = data;
    value.v1.clientBuf.dataSize = bytes;
    return value;
}

bool ok(Qnn_ErrorHandle_t status, Error* error, const char* stage) {
    if (status == QNN_SUCCESS) return true;
    fail(error, stage, status);
    return false;
}

bool add_matmul(Linear* linear, const char* op_name, const Qnn_Tensor_t& input,
                const Qnn_Tensor_t& weight, const Qnn_Tensor_t& output, Error* error) {
    std::array<Qnn_Tensor_t, 2> inputs = {input, weight};
    std::array<Qnn_Tensor_t, 1> outputs = {output};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = op_name;
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = QNN_OP_MAT_MUL;
    op.v1.numOfInputs = inputs.size();
    op.v1.inputTensors = inputs.data();
    op.v1.numOfOutputs = outputs.size();
    op.v1.outputTensors = outputs.data();
    return ok(linear->api.graphAddNode(linear->graph, op), error, "QnnGraph_addNode(MatMul)");
}
} // namespace

static void* create_linear(const char* backend_path, uint32_t rows, uint32_t inner,
                           uint32_t columns, const int8_t* weights, const float* scales,
                           uint32_t columns2, const int8_t* weights2, const float* scales2,
                           uint32_t columns3, const int8_t* weights3, const float* scales3,
                           uint32_t weight_bits, float baked_input_scale,
                           float output_scale, float output2_scale, float output3_scale,
                           Error* error) {
    auto* linear = new (std::nothrow) Linear();
    if (linear == nullptr) { fail(error, "分配 QNN Linear"); return nullptr; }
    const auto graph_id = next_graph_id.fetch_add(1);
    {
        std::lock_guard<std::recursive_mutex> lock(shared_mutex);
        if (shared_users == 0) {
            shared_library = dlopen(backend_path, RTLD_NOW | RTLD_GLOBAL);
            if (shared_library == nullptr) { fail(error, dlerror()); delete linear; return nullptr; }
            auto providers_fn = reinterpret_cast<GetProviders>(dlsym(shared_library, "QnnInterface_getProviders"));
            if (providers_fn == nullptr) { fail(error, "找不到 QnnInterface_getProviders"); dlclose(shared_library); shared_library = nullptr; delete linear; return nullptr; }
            const QnnInterface_t** providers = nullptr;
            uint32_t count = 0;
            if (!ok(providers_fn(&providers, &count), error, "QnnInterface_getProviders") || providers == nullptr) { dlclose(shared_library); shared_library = nullptr; delete linear; return nullptr; }
            bool found = false;
            for (uint32_t i = 0; i < count; ++i) {
                const auto version = providers[i]->apiVersion.coreApiVersion;
                if (version.major == QNN_API_VERSION_MAJOR && version.minor >= QNN_API_VERSION_MINOR) { shared_api = providers[i]->QNN_INTERFACE_VER_NAME; found = true; break; }
            }
            if (!found) { fail(error, "没有兼容的 QNN core interface"); dlclose(shared_library); shared_library = nullptr; delete linear; return nullptr; }
            if (!ok(shared_api.backendCreate(nullptr, nullptr, &shared_backend), error, "QnnBackend_create")) { dlclose(shared_library); shared_library = nullptr; delete linear; return nullptr; }
            if (shared_api.deviceCreate != nullptr) {
                auto status = shared_api.deviceCreate(nullptr, nullptr, &shared_device);
                if (status == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) shared_device = nullptr;
                else if (!ok(status, error, "QnnDevice_create")) { shared_api.backendFree(shared_backend); dlclose(shared_library); shared_library = nullptr; shared_backend = nullptr; delete linear; return nullptr; }
            }
            if (shared_api.logCreate != nullptr && getenv("ZLLM_QNN_LOG") != nullptr) {
                Qnn_LogHandle_t log_handle = nullptr;
                shared_api.logCreate(log_callback, strcmp(getenv("ZLLM_QNN_LOG"), "2") == 0 ? QNN_LOG_LEVEL_ERROR : QNN_LOG_LEVEL_DEBUG, &log_handle);
            }
            for (auto& context : shared_contexts) {
                if (!ok(shared_api.contextCreate(shared_backend, shared_device, nullptr, &context), error, "QnnContext_create")) { delete linear; return nullptr; }
            }
        }
        ++shared_users;
        linear->library = shared_library; linear->api = shared_api; linear->backend = shared_backend;
        // 诊断实验(ZLLM_QNN_SINGLE_CTX=1):全部图进 ctx0,隔离双 context 交替切换开销。
        const bool single_ctx = getenv("ZLLM_QNN_SINGLE_CTX") != nullptr;
        linear->device = shared_device; linear->context = shared_contexts[single_ctx ? 0 : (graph_id / 2) % shared_contexts.size()];
    }
    const auto suffix = std::to_string(graph_id);
    linear->graph_name = "zllm_linear_" + suffix;
    linear->input_name = "input_" + suffix; linear->weight_name = "weight_" + suffix;
    linear->output_name = "output_" + suffix; linear->weight2_name = "weight2_" + suffix;
    linear->output2_name = "output2_" + suffix; linear->op_name = "linear_" + suffix;
    linear->op2_name = "linear2_" + suffix;
    linear->weight3_name = "weight3_" + suffix; linear->output3_name = "output3_" + suffix;
    linear->op3_name = "linear3_" + suffix;
    if (!ok(linear->api.graphCreate(linear->context, linear->graph_name.c_str(), nullptr, &linear->graph), error, "QnnGraph_create")) { delete linear; return nullptr; }

    linear->input_dims = {rows, inner};
    linear->weight_dims = {inner, columns};
    linear->output_dims = {rows, columns};
    linear->weights.assign(weights, weights + static_cast<size_t>(inner) * columns);
    linear->scales.assign(scales, scales + columns);
    linear->scale_offsets.resize(columns);
    for (uint32_t n = 0; n < columns; ++n) linear->scale_offsets[n] = {linear->scales[n], 0};
    linear->dual = weights2 != nullptr && scales2 != nullptr;
    if (linear->dual) {
        linear->weight2_dims = {inner, columns2};
        linear->output2_dims = {rows, columns2};
        linear->weights2.assign(weights2, weights2 + static_cast<size_t>(inner) * columns2);
        linear->scales2.assign(scales2, scales2 + columns2);
        linear->scale_offsets2.resize(columns2);
        for (uint32_t n = 0; n < columns2; ++n) linear->scale_offsets2[n] = {linear->scales2[n], 0};
    }
    const bool triple = weights3 != nullptr && scales3 != nullptr;
    if (triple) {
        linear->weight3_dims = {inner, columns3};
        linear->output3_dims = {rows, columns3};
        linear->weights3.assign(weights3, weights3 + static_cast<size_t>(inner) * columns3);
        linear->scales3.assign(scales3, scales3 + columns3);
        linear->scale_offsets3.resize(columns3);
        for (uint32_t n = 0; n < columns3; ++n) linear->scale_offsets3[n] = {linear->scales3[n], 0};
    }

    // 图内烘焙输入 scale:真实输入 scale 由客户端逐行动态决定,读回按比值校正。
    Qnn_QuantizeParams_t input_quant = QNN_QUANTIZE_PARAMS_INIT;
    input_quant.encodingDefinition = QNN_DEFINITION_DEFINED;
    input_quant.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    input_quant.scaleOffsetEncoding.scale = baked_input_scale;
    input_quant.scaleOffsetEncoding.offset = 0;
    Qnn_QuantizeParams_t weight_quant = QNN_QUANTIZE_PARAMS_INIT;
    weight_quant.encodingDefinition = QNN_DEFINITION_DEFINED;
    if (weight_bits == 4) {
        weight_quant.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
        weight_quant.bwAxisScaleOffsetEncoding.bitwidth = 4;
        weight_quant.bwAxisScaleOffsetEncoding.axis = 1;
        weight_quant.bwAxisScaleOffsetEncoding.numElements = columns;
        weight_quant.bwAxisScaleOffsetEncoding.scales = linear->scales.data();
        weight_quant.bwAxisScaleOffsetEncoding.offsets = nullptr;
    } else {
        weight_quant.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
        weight_quant.axisScaleOffsetEncoding.axis = 1;
        weight_quant.axisScaleOffsetEncoding.numScaleOffsets = columns;
        weight_quant.axisScaleOffsetEncoding.scaleOffset = linear->scale_offsets.data();
    }
    Qnn_QuantizeParams_t output_quant = QNN_QUANTIZE_PARAMS_INIT;
    output_quant.encodingDefinition = QNN_DEFINITION_DEFINED;
    output_quant.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    output_quant.scaleOffsetEncoding.scale = output_scale;
    output_quant.scaleOffsetEncoding.offset = 0;
    Qnn_QuantizeParams_t output2_quant = output_quant;
    output2_quant.scaleOffsetEncoding.scale = output2_scale;
    Qnn_QuantizeParams_t output3_quant = output_quant;
    output3_quant.scaleOffsetEncoding.scale = output3_scale;

    linear->input = quantized_tensor(linear->input_name.c_str(), QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_SFIXED_POINT_8, input_quant, linear->input_dims.data(), nullptr, 0);
    auto weight = quantized_tensor(linear->weight_name.c_str(), QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_SFIXED_POINT_8, weight_quant, linear->weight_dims.data(), linear->weights.data(), static_cast<uint32_t>(linear->weights.size()));
    linear->output = quantized_tensor(linear->output_name.c_str(), QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_SFIXED_POINT_8, output_quant, linear->output_dims.data(), nullptr, 0);
    if (!ok(linear->api.tensorCreateGraphTensor(linear->graph, &linear->input), error, "QnnTensor_create(input)") ||
        !ok(linear->api.tensorCreateGraphTensor(linear->graph, &weight), error, "QnnTensor_create(weight)") ||
        !ok(linear->api.tensorCreateGraphTensor(linear->graph, &linear->output), error, "QnnTensor_create(output)") ||
        !add_matmul(linear, linear->op_name.c_str(), linear->input, weight, linear->output, error)) { delete linear; return nullptr; }
    if (linear->dual) {
        Qnn_QuantizeParams_t weight2_quant = weight_quant;
        if (weight_bits == 4) {
            weight2_quant.bwAxisScaleOffsetEncoding.numElements = columns2;
            weight2_quant.bwAxisScaleOffsetEncoding.scales = linear->scales2.data();
        } else {
            weight2_quant.axisScaleOffsetEncoding.numScaleOffsets = columns2;
            weight2_quant.axisScaleOffsetEncoding.scaleOffset = linear->scale_offsets2.data();
        }
        auto weight2 = quantized_tensor(linear->weight2_name.c_str(), QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_SFIXED_POINT_8, weight2_quant, linear->weight2_dims.data(), linear->weights2.data(), static_cast<uint32_t>(linear->weights2.size()));
        linear->output2 = quantized_tensor(linear->output2_name.c_str(), QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_SFIXED_POINT_8, output2_quant, linear->output2_dims.data(), nullptr, 0);
        if (!ok(linear->api.tensorCreateGraphTensor(linear->graph, &weight2), error, "QnnTensor_create(weight2)") ||
            !ok(linear->api.tensorCreateGraphTensor(linear->graph, &linear->output2), error, "QnnTensor_create(output2)") ||
            !add_matmul(linear, linear->op2_name.c_str(), linear->input, weight2, linear->output2, error)) { delete linear; return nullptr; }
    }
    if (triple) {
        Qnn_QuantizeParams_t weight3_quant = weight_quant;
        if (weight_bits == 4) {
            weight3_quant.bwAxisScaleOffsetEncoding.numElements = columns3;
            weight3_quant.bwAxisScaleOffsetEncoding.scales = linear->scales3.data();
        } else {
            weight3_quant.axisScaleOffsetEncoding.numScaleOffsets = columns3;
            weight3_quant.axisScaleOffsetEncoding.scaleOffset = linear->scale_offsets3.data();
        }
        auto weight3 = quantized_tensor(linear->weight3_name.c_str(), QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_SFIXED_POINT_8, weight3_quant, linear->weight3_dims.data(), linear->weights3.data(), static_cast<uint32_t>(linear->weights3.size()));
        linear->output3 = quantized_tensor(linear->output3_name.c_str(), QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_SFIXED_POINT_8, output3_quant, linear->output3_dims.data(), nullptr, 0);
        if (!ok(linear->api.tensorCreateGraphTensor(linear->graph, &weight3), error, "QnnTensor_create(weight3)") ||
            !ok(linear->api.tensorCreateGraphTensor(linear->graph, &linear->output3), error, "QnnTensor_create(output3)") ||
            !add_matmul(linear, linear->op3_name.c_str(), linear->input, weight3, linear->output3, error)) { delete linear; return nullptr; }
    }
    const auto finalize_status = linear->api.graphFinalize(linear->graph, nullptr, nullptr);
    if (finalize_status != QNN_SUCCESS) {
        char stage[160];
        std::snprintf(stage, sizeof(stage), "QnnGraph_finalize graph=%s M=%u K=%u N=%u bits=%u", linear->graph_name.c_str(), rows, inner, columns, weight_bits);
        fail(error, stage, finalize_status);
        delete linear;
        return nullptr;
    }
    if (linear->api.contextGetBinarySize != nullptr) linear->api.contextGetBinarySize(linear->context, &linear->context_bytes);
    return linear;
}

extern "C" void* zllm_qnn_linear_create(const char* backend_path, uint32_t rows, uint32_t inner, uint32_t columns,
                                        const int8_t* weights, const float* scales, uint32_t weight_bits,
                                        float baked_input_scale, float output_scale, Error* error) {
    return create_linear(backend_path, rows, inner, columns, weights, scales, 0, nullptr, nullptr, 0, nullptr, nullptr,
                         weight_bits, baked_input_scale, output_scale, output_scale, output_scale, error);
}

extern "C" void* zllm_qnn_dual_linear_create(const char* backend_path, uint32_t rows, uint32_t inner,
                                             uint32_t columns, const int8_t* weights, const float* scales,
                                             uint32_t columns2, const int8_t* weights2, const float* scales2,
                                             uint32_t weight_bits, float baked_input_scale,
                                             float output_scale, float output2_scale, Error* error) {
    return create_linear(backend_path, rows, inner, columns, weights, scales, columns2, weights2, scales2, 0, nullptr, nullptr,
                         weight_bits, baked_input_scale, output_scale, output2_scale, output2_scale, error);
}

extern "C" void* zllm_qnn_triple_linear_create(const char* backend_path, uint32_t rows, uint32_t inner,
                                               uint32_t columns, const int8_t* weights, const float* scales,
                                               uint32_t columns2, const int8_t* weights2, const float* scales2,
                                               uint32_t columns3, const int8_t* weights3, const float* scales3,
                                               uint32_t weight_bits, float baked_input_scale,
                                               float output_scale, float output2_scale, float output3_scale, Error* error) {
    return create_linear(backend_path, rows, inner, columns, weights, scales, columns2, weights2, scales2, columns3, weights3, scales3,
                         weight_bits, baked_input_scale, output_scale, output2_scale, output3_scale, error);
}


// 共享 backend/device/context 初始化;成功后 shared_users 已计入本实例。
static bool shared_init(const char* backend_path, Error* error) {
    std::lock_guard<std::recursive_mutex> lock(shared_mutex);
    if (shared_users == 0) {
        shared_library = dlopen(backend_path, RTLD_NOW | RTLD_GLOBAL);
        if (shared_library == nullptr) { fail(error, dlerror()); return false; }
        auto providers_fn = reinterpret_cast<GetProviders>(dlsym(shared_library, "QnnInterface_getProviders"));
        if (providers_fn == nullptr) { fail(error, "找不到 QnnInterface_getProviders"); dlclose(shared_library); shared_library = nullptr; return false; }
        const QnnInterface_t** providers = nullptr;
        uint32_t count = 0;
        if (!ok(providers_fn(&providers, &count), error, "QnnInterface_getProviders") || providers == nullptr) { dlclose(shared_library); shared_library = nullptr; return false; }
        bool found = false;
        for (uint32_t i = 0; i < count; ++i) {
            const auto version = providers[i]->apiVersion.coreApiVersion;
            if (version.major == QNN_API_VERSION_MAJOR && version.minor >= QNN_API_VERSION_MINOR) { shared_api = providers[i]->QNN_INTERFACE_VER_NAME; found = true; break; }
        }
        if (!found) { fail(error, "没有兼容的 QNN core interface"); dlclose(shared_library); shared_library = nullptr; return false; }
        if (!ok(shared_api.backendCreate(nullptr, nullptr, &shared_backend), error, "QnnBackend_create")) { dlclose(shared_library); shared_library = nullptr; return false; }
        if (shared_api.deviceCreate != nullptr) {
            auto status = shared_api.deviceCreate(nullptr, nullptr, &shared_device);
            if (status == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) shared_device = nullptr;
            else if (!ok(status, error, "QnnDevice_create")) { shared_api.backendFree(shared_backend); dlclose(shared_library); shared_library = nullptr; shared_backend = nullptr; return false; }
        }
        for (auto& context : shared_contexts) {
            if (!ok(shared_api.contextCreate(shared_backend, shared_device, nullptr, &context), error, "QnnContext_create")) return false;
        }
        if (shared_api.logCreate != nullptr && getenv("ZLLM_QNN_LOG") != nullptr) {
            Qnn_LogHandle_t log_handle = nullptr;
            shared_api.logCreate(log_callback, strcmp(getenv("ZLLM_QNN_LOG"), "2") == 0 ? QNN_LOG_LEVEL_ERROR : QNN_LOG_LEVEL_DEBUG, &log_handle);
        }
    }
    ++shared_users;
    return true;
}

// ── 量化算子探针 ──
// 复用 Linear 持有共享 backend/device/context;探针图 1 进 1 出(乘法 2 进),S8。
static void* create_elementwise(const char* backend_path, const char* op_type,
                                const char* graph_name, uint32_t count,
                                const int8_t* gamma, float gamma_scale,
                                float input_scale, float output_scale, Error* error) {
    auto* linear = new (std::nothrow) Linear();
    if (linear == nullptr) { fail(error, "分配探针"); return nullptr; }
    const auto graph_id = next_graph_id.fetch_add(1);
    {
        if (!shared_init(backend_path, error)) { delete linear; return nullptr; }
        linear->api = shared_api; linear->context = shared_contexts[(graph_id / 2) % shared_contexts.size()];
    }
    if (!ok(linear->api.graphCreate(linear->context, graph_name, nullptr, &linear->graph), error, "QnnGraph_create")) { delete linear; return nullptr; }

    linear->input_dims = {1, count};
    linear->output_dims = {1, count};
    linear->input_name = "probe_in"; linear->output_name = "probe_out";
    Qnn_QuantizeParams_t in_quant = QNN_QUANTIZE_PARAMS_INIT;
    in_quant.encodingDefinition = QNN_DEFINITION_DEFINED;
    in_quant.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    in_quant.scaleOffsetEncoding.scale = input_scale;
    in_quant.scaleOffsetEncoding.offset = 0;
    Qnn_QuantizeParams_t out_quant = in_quant;
    out_quant.scaleOffsetEncoding.scale = output_scale;

    linear->input = quantized_tensor(linear->input_name.c_str(), QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_SFIXED_POINT_8, in_quant, linear->input_dims.data(), nullptr, 0);
    linear->output = quantized_tensor(linear->output_name.c_str(), QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_SFIXED_POINT_8, out_quant, linear->output_dims.data(), nullptr, 0);
    if (!ok(linear->api.tensorCreateGraphTensor(linear->graph, &linear->input), error, "QnnTensor_create(probe_in)") ||
        !ok(linear->api.tensorCreateGraphTensor(linear->graph, &linear->output), error, "QnnTensor_create(probe_out)")) { delete linear; return nullptr; }

    std::array<Qnn_Tensor_t, 2> op_inputs = {linear->input, linear->input};
    std::array<Qnn_Tensor_t, 1> op_outputs = {linear->output};
    uint32_t num_inputs = 1;
    if (gamma != nullptr) {
        // RmsNorm 的 gamma:S8 per-channel 对称量化(scale = gamma_scale)。
        linear->scale_offsets.assign(count, {gamma_scale, 0});
        Qnn_QuantizeParams_t gamma_quant = QNN_QUANTIZE_PARAMS_INIT;
        gamma_quant.encodingDefinition = QNN_DEFINITION_DEFINED;
        gamma_quant.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
        gamma_quant.axisScaleOffsetEncoding.axis = 1;
        gamma_quant.axisScaleOffsetEncoding.numScaleOffsets = count;
        gamma_quant.axisScaleOffsetEncoding.scaleOffset = linear->scale_offsets.data();
        linear->weights.assign(gamma, gamma + count);
        linear->weight_dims = {1, count};
        auto gamma_tensor = quantized_tensor("probe_gamma", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_SFIXED_POINT_8, gamma_quant, linear->weight_dims.data(), linear->weights.data(), count);
        if (!ok(linear->api.tensorCreateGraphTensor(linear->graph, &gamma_tensor), error, "QnnTensor_create(gamma)")) { delete linear; return nullptr; }
        op_inputs[1] = gamma_tensor;
        num_inputs = 2;
    } else if (strcmp(op_type, QNN_OP_ELEMENT_WISE_MULTIPLY) == 0) {
        num_inputs = 2;  // 同一输入张量乘自身(探针语义:平方)
    }

    Qnn_Param_t params[2] = {QNN_PARAM_INIT, QNN_PARAM_INIT};
    uint32_t param_count = 0;
    int32_t axes_value = 1;
    if (strcmp(op_type, QNN_OP_RMS_NORM) == 0) {
        params[0].paramType = QNN_PARAMTYPE_SCALAR;
        params[0].name = QNN_OP_RMS_NORM_PARAM_EPSILON;
        params[0].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
        params[0].scalarParam.floatValue = 1.0e-6f;
        linear->int32_scratch = axes_value;
        params[1].paramType = QNN_PARAMTYPE_TENSOR;
        params[1].name = QNN_OP_RMS_NORM_PARAM_AXES;
        params[1].tensorParam = quantized_tensor("probe_axes", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, QNN_QUANTIZE_PARAMS_INIT, linear->axes_dims.data(), &linear->int32_scratch, 4);
        if (!ok(linear->api.tensorCreateGraphTensor(linear->graph, &params[1].tensorParam), error, "QnnTensor_create(axes)")) { delete linear; return nullptr; }
        param_count = 2;
    }
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = "probe_op";
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = op_type;
    op.v1.numOfInputs = num_inputs;
    op.v1.inputTensors = op_inputs.data();
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = op_outputs.data();
    op.v1.numOfParams = param_count;
    op.v1.params = params;
    if (!ok(linear->api.graphAddNode(linear->graph, op), error, "QnnGraph_addNode(probe)")) { delete linear; return nullptr; }

    const auto finalize_status = linear->api.graphFinalize(linear->graph, nullptr, nullptr);
    if (finalize_status != QNN_SUCCESS) {
        char stage[160];
        std::snprintf(stage, sizeof(stage), "QnnGraph_finalize probe op=%s", op_type);
        fail(error, stage, finalize_status);
        delete linear;
        return nullptr;
    }
    return linear;
}

extern "C" void* zllm_qnn_probe_sigmoid_create(const char* backend, uint32_t count, float input_scale, float output_scale, Error* error) {
    return create_elementwise(backend, QNN_OP_SIGMOID, "zllm_probe_sigmoid", count, nullptr, 1.0f, input_scale, output_scale, error);
}

extern "C" void* zllm_qnn_probe_multiply_create(const char* backend, uint32_t count, float input_scale, float output_scale, Error* error) {
    return create_elementwise(backend, QNN_OP_ELEMENT_WISE_MULTIPLY, "zllm_probe_multiply", count, nullptr, 1.0f, input_scale, output_scale, error);
}

extern "C" void* zllm_qnn_probe_rmsnorm_create(const char* backend, uint32_t count, const int8_t* gamma, float gamma_scale, float input_scale, float output_scale, Error* error) {
    return create_elementwise(backend, QNN_OP_RMS_NORM, "zllm_probe_rmsnorm", count, gamma, gamma_scale, input_scale, output_scale, error);
}

extern "C" int zllm_qnn_probe_execute(void* handle, const int8_t* input, int8_t* output, Error* error) {
    auto* linear = static_cast<Linear*>(handle);
    if (linear == nullptr) { fail(error, "探针 handle 为空"); return -1; }
    auto input_tensor = linear->input;
    input_tensor.v1.clientBuf.data = const_cast<int8_t*>(input);
    input_tensor.v1.clientBuf.dataSize = linear->input_dims[1];
    auto output_tensor = linear->output;
    output_tensor.v1.clientBuf.data = output;
    output_tensor.v1.clientBuf.dataSize = linear->output_dims[1];
    return ok(linear->api.graphExecute(linear->graph, &input_tensor, 1, &output_tensor, 1, nullptr, nullptr), error, "QnnGraph_execute(probe)") ? 0 : -1;
}

static int execute_linear(Linear* linear, const int8_t* input, int8_t* output, int8_t* output2, int8_t* output3, Error* error) {
    if (linear == nullptr) { fail(error, "QNN Linear handle 为空"); return -1; }
    auto input_tensor = linear->input;
    input_tensor.v1.clientBuf.data = const_cast<int8_t*>(input);
    input_tensor.v1.clientBuf.dataSize = linear->input_dims[0] * linear->input_dims[1];
    auto first = linear->output;
    first.v1.clientBuf.data = output;
    first.v1.clientBuf.dataSize = linear->output_dims[1];
    std::array<Qnn_Tensor_t, 3> outputs = {first, linear->output2, linear->output3};
    uint32_t count = 1;
    if (output2 != nullptr) {
        outputs[1].v1.clientBuf.data = output2;
        outputs[1].v1.clientBuf.dataSize = linear->output2_dims[1];
        count = 2;
    }
    if (output3 != nullptr) {
        outputs[2].v1.clientBuf.data = output3;
        outputs[2].v1.clientBuf.dataSize = linear->output3_dims[1];
        count = 3;
    }
    const auto status = linear->api.graphExecute(linear->graph, &input_tensor, 1, outputs.data(), count, nullptr, nullptr);
    if (status != QNN_SUCCESS) {
        char stage[192];
        std::snprintf(stage, sizeof(stage), "QnnGraph_execute graph=%s K=%u N=%u", linear->graph_name.c_str(), linear->input_dims[1], linear->output_dims[1]);
        fail(error, stage, status);
        return -1;
    }
    return 0;
}

extern "C" int zllm_qnn_linear_execute(void* handle, const int8_t* input, int8_t* output, Error* error) {
    return execute_linear(static_cast<Linear*>(handle), input, output, nullptr, nullptr, error);
}

extern "C" int zllm_qnn_dual_linear_execute(void* handle, const int8_t* input, int8_t* output, int8_t* output2, Error* error) {
    return execute_linear(static_cast<Linear*>(handle), input, output, output2, nullptr, error);
}

extern "C" int zllm_qnn_triple_linear_execute(void* handle, const int8_t* input, int8_t* output, int8_t* output2, int8_t* output3, Error* error) {
    return execute_linear(static_cast<Linear*>(handle), input, output, output2, output3, error);
}


// ── MLP 整段融合图:in → MatMul×w_gate → g;MatMul×w_up → u;
//    Sigmoid(g) → s;Multiply(s,g) → silu;Multiply(silu,u) → act;MatMul×w_down → out ──
struct MlpGraph {
    QNN_INTERFACE_VER_TYPE api{};
    Qnn_GraphHandle_t graph = nullptr;
    Qnn_Tensor_t input = QNN_TENSOR_INIT;
    Qnn_Tensor_t output = QNN_TENSOR_INIT;
    std::array<uint32_t, 2> input_dims{};
    std::array<uint32_t, 2> output_dims{};
    uint32_t intermediate = 0;
    float baked_input_scale = 1.0f;
    float output_scale = 1.0f;
    std::vector<int8_t> weights_gate, weights_up, weights_down;
    std::vector<float> scales_gate, scales_up, scales_down;
    std::vector<Qnn_ScaleOffset_t> so_gate, so_up, so_down;
    std::vector<std::string> names;

    ~MlpGraph() {
        std::lock_guard<std::recursive_mutex> lock(shared_mutex);
        if (shared_users == 0 || --shared_users != 0) return;
        if (shared_api.contextFree != nullptr) for (auto context : shared_contexts) if (context != nullptr) shared_api.contextFree(context, nullptr);
        if (shared_device != nullptr && shared_api.deviceFree != nullptr) shared_api.deviceFree(shared_device);
        if (shared_backend != nullptr && shared_api.backendFree != nullptr) shared_api.backendFree(shared_backend);
        if (shared_library != nullptr) dlclose(shared_library);
        shared_library = nullptr; shared_backend = nullptr; shared_device = nullptr; shared_contexts = {};
    }
};

static void fill_axis_encoding(Qnn_QuantizeParams_t& quant, std::vector<Qnn_ScaleOffset_t>& so, const std::vector<float>& scales, uint32_t columns) {
    so.resize(columns);
    for (uint32_t n = 0; n < columns; ++n) so[n] = {scales[n], 0};
    quant.encodingDefinition = QNN_DEFINITION_DEFINED;
    quant.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
    quant.axisScaleOffsetEncoding.axis = 1;
    quant.axisScaleOffsetEncoding.numScaleOffsets = columns;
    quant.axisScaleOffsetEncoding.scaleOffset = so.data();
}

extern "C" void* zllm_qnn_mlp_create(const char* backend_path, uint32_t rows, uint32_t inner,
                                     uint32_t gate_columns, const int8_t* gate_weights, const float* gate_scales,
                                     uint32_t up_columns, const int8_t* up_weights, const float* up_scales,
                                     uint32_t down_columns, const int8_t* down_weights, const float* down_scales,
                                     uint32_t weight_bits,
                                     float baked_input_scale,
                                     float gate_out_scale, float up_out_scale,
                                     float silu_scale, float act_scale, float out_scale,
                                     Error* error) {
    auto* mlp = new (std::nothrow) MlpGraph();
    if (mlp == nullptr) { fail(error, "分配 MLP 图"); return nullptr; }
    const auto graph_id = next_graph_id.fetch_add(1);
    if (!shared_init(backend_path, error)) { delete mlp; return nullptr; }
    mlp->api = shared_api;
    mlp->intermediate = gate_columns;
    mlp->baked_input_scale = baked_input_scale;
    mlp->output_scale = out_scale;
    mlp->input_dims = {rows, inner};
    mlp->output_dims = {rows, down_columns};
    mlp->weights_gate.assign(gate_weights, gate_weights + static_cast<size_t>(inner) * gate_columns);
    mlp->scales_gate.assign(gate_scales, gate_scales + gate_columns);
    mlp->weights_up.assign(up_weights, up_weights + static_cast<size_t>(inner) * up_columns);
    mlp->scales_up.assign(up_scales, up_scales + up_columns);
    mlp->weights_down.assign(down_weights, down_weights + static_cast<size_t>(gate_columns) * down_columns);
    mlp->scales_down.assign(down_scales, down_scales + down_columns);

    const auto suffix = std::to_string(graph_id);
    const auto graph_name = "zllm_mlp_" + suffix;
    Qnn_ContextHandle_t context = shared_contexts[(graph_id / 2) % shared_contexts.size()];
    if (!ok(mlp->api.graphCreate(context, graph_name.c_str(), nullptr, &mlp->graph), error, "QnnGraph_create(mlp)")) { delete mlp; return nullptr; }

    auto tensor = [](const std::string& name, Qnn_TensorType_t type, Qnn_DataType_t dtype, Qnn_QuantizeParams_t quant, uint32_t* dims, void* data, uint32_t bytes) {
        Qnn_Tensor_t value = QNN_TENSOR_INIT;
        value.version = QNN_TENSOR_VERSION_1;
        value.v1.name = name.c_str();
        value.v1.type = type;
        value.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        value.v1.dataType = dtype;
        value.v1.quantizeParams = quant;
        value.v1.rank = 2;
        value.v1.dimensions = dims;
        value.v1.memType = QNN_TENSORMEMTYPE_RAW;
        value.v1.clientBuf.data = data;
        value.v1.clientBuf.dataSize = bytes;
        return value;
    };
    auto scale_offset = [](float scale) {
        Qnn_QuantizeParams_t quant = QNN_QUANTIZE_PARAMS_INIT;
        quant.encodingDefinition = QNN_DEFINITION_DEFINED;
        quant.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        quant.scaleOffsetEncoding.scale = scale;
        quant.scaleOffsetEncoding.offset = 0;
        return quant;
    };

    std::array<uint32_t, 2> gate_dims = {inner, gate_columns};
    std::array<uint32_t, 2> up_dims = {inner, up_columns};
    std::array<uint32_t, 2> down_dims = {gate_columns, down_columns};
    std::array<uint32_t, 2> mid_dims = {rows, gate_columns};

    // 4-bit 权重走 BW_AXIS,8-bit 走 AXIS_SCALE_OFFSET(2.38 校验要求)
    auto weight_quant = [&](std::vector<Qnn_ScaleOffset_t>& so, const std::vector<float>& scales, uint32_t columns) {
        Qnn_QuantizeParams_t quant = QNN_QUANTIZE_PARAMS_INIT;
        quant.encodingDefinition = QNN_DEFINITION_DEFINED;
        if (weight_bits == 4) {
            quant.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
            quant.bwAxisScaleOffsetEncoding.bitwidth = 4;
            quant.bwAxisScaleOffsetEncoding.axis = 1;
            quant.bwAxisScaleOffsetEncoding.numElements = columns;
            quant.bwAxisScaleOffsetEncoding.scales = const_cast<float*>(scales.data());
            quant.bwAxisScaleOffsetEncoding.offsets = nullptr;
        } else {
            fill_axis_encoding(quant, so, scales, columns);
        }
        return quant;
    };

    mlp->input = tensor("mlp_in_" + suffix, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_SFIXED_POINT_8, scale_offset(baked_input_scale), mlp->input_dims.data(), nullptr, 0);
    mlp->output = tensor("mlp_out_" + suffix, QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_SFIXED_POINT_8, scale_offset(out_scale), mlp->output_dims.data(), nullptr, 0);
    auto w_gate = tensor("mlp_wg_" + suffix, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_SFIXED_POINT_8, weight_quant(mlp->so_gate, mlp->scales_gate, gate_columns), gate_dims.data(), mlp->weights_gate.data(), static_cast<uint32_t>(mlp->weights_gate.size()));
    auto w_up = tensor("mlp_wu_" + suffix, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_SFIXED_POINT_8, weight_quant(mlp->so_up, mlp->scales_up, up_columns), up_dims.data(), mlp->weights_up.data(), static_cast<uint32_t>(mlp->weights_up.size()));
    auto w_down = tensor("mlp_wd_" + suffix, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_SFIXED_POINT_8, weight_quant(mlp->so_down, mlp->scales_down, down_columns), down_dims.data(), mlp->weights_down.data(), static_cast<uint32_t>(mlp->weights_down.size()));
    auto t_gate = tensor("mlp_g_" + suffix, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_SFIXED_POINT_8, scale_offset(gate_out_scale), mid_dims.data(), nullptr, 0);
    auto t_up = tensor("mlp_u_" + suffix, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_SFIXED_POINT_8, scale_offset(up_out_scale), mid_dims.data(), nullptr, 0);
    auto t_sig = tensor("mlp_s_" + suffix, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_SFIXED_POINT_8, scale_offset(1.0f / 128.0f), mid_dims.data(), nullptr, 0);
    auto t_silu = tensor("mlp_sg_" + suffix, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_SFIXED_POINT_8, scale_offset(silu_scale), mid_dims.data(), nullptr, 0);
    auto t_act = tensor("mlp_a_" + suffix, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_SFIXED_POINT_8, scale_offset(act_scale), mid_dims.data(), nullptr, 0);

    Qnn_Tensor_t* tensors[] = {&mlp->input, &mlp->output, &w_gate, &w_up, &w_down, &t_gate, &t_up, &t_sig, &t_silu, &t_act};
    for (auto* t : tensors) {
        char stage[192];
        std::snprintf(stage, sizeof(stage), "QnnTensor_create(mlp %s type=%d dtype=0x%x)", t->v1.name, static_cast<int>(t->v1.type), static_cast<int>(t->v1.dataType));
        if (!ok(mlp->api.tensorCreateGraphTensor(mlp->graph, t), error, stage)) { delete mlp; return nullptr; }
    }

    auto add_node = [&](const std::string& name, const char* type, Qnn_Tensor_t* inputs, uint32_t in_count, Qnn_Tensor_t* outputs, uint32_t out_count) {
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.v1.name = name.c_str();
        op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        op.v1.typeName = type;
        op.v1.numOfInputs = in_count;
        op.v1.inputTensors = inputs;
        op.v1.numOfOutputs = out_count;
        op.v1.outputTensors = outputs;
        return ok(mlp->api.graphAddNode(mlp->graph, op), error, "QnnGraph_addNode(mlp)");
    };

    std::array<Qnn_Tensor_t, 2> mm_gate_in = {mlp->input, w_gate};
    std::array<Qnn_Tensor_t, 2> mm_up_in = {mlp->input, w_up};
    std::array<Qnn_Tensor_t, 2> mm_down_in = {t_act, w_down};
    std::array<Qnn_Tensor_t, 2> mul_silu_in = {t_sig, t_gate};
    std::array<Qnn_Tensor_t, 2> mul_act_in = {t_silu, t_up};
    if (!add_node("mlp_mm_g_" + suffix, QNN_OP_MAT_MUL, mm_gate_in.data(), 2, &t_gate, 1) ||
        !add_node("mlp_mm_u_" + suffix, QNN_OP_MAT_MUL, mm_up_in.data(), 2, &t_up, 1) ||
        !add_node("mlp_sig_" + suffix, QNN_OP_SIGMOID, &t_gate, 1, &t_sig, 1) ||
        !add_node("mlp_mul_silu_" + suffix, QNN_OP_ELEMENT_WISE_MULTIPLY, mul_silu_in.data(), 2, &t_silu, 1) ||
        !add_node("mlp_mul_act_" + suffix, QNN_OP_ELEMENT_WISE_MULTIPLY, mul_act_in.data(), 2, &t_act, 1) ||
        !add_node("mlp_mm_d_" + suffix, QNN_OP_MAT_MUL, mm_down_in.data(), 2, &mlp->output, 1)) { delete mlp; return nullptr; }

    const auto finalize_status = mlp->api.graphFinalize(mlp->graph, nullptr, nullptr);
    if (finalize_status != QNN_SUCCESS) {
        char stage[160];
        std::snprintf(stage, sizeof(stage), "QnnGraph_finalize mlp K=%u N=%u", inner, down_columns);
        fail(error, stage, finalize_status);
        delete mlp;
        return nullptr;
    }
    return mlp;
}

extern "C" int zllm_qnn_mlp_execute(void* handle, const int8_t* input, int8_t* output, Error* error) {
    auto* mlp = static_cast<MlpGraph*>(handle);
    if (mlp == nullptr) { fail(error, "MLP 图 handle 为空"); return -1; }
    auto input_tensor = mlp->input;
    input_tensor.v1.clientBuf.data = const_cast<int8_t*>(input);
    input_tensor.v1.clientBuf.dataSize = mlp->input_dims[1];
    auto output_tensor = mlp->output;
    output_tensor.v1.clientBuf.data = output;
    output_tensor.v1.clientBuf.dataSize = mlp->output_dims[1];
    return ok(mlp->api.graphExecute(mlp->graph, &input_tensor, 1, &output_tensor, 1, nullptr, nullptr), error, "QnnGraph_execute(mlp)") ? 0 : -1;
}

extern "C" void zllm_qnn_mlp_destroy(void* handle) { delete static_cast<MlpGraph*>(handle); }

extern "C" uint64_t zllm_qnn_linear_context_bytes(const void* handle) {
    const auto* linear = static_cast<const Linear*>(handle);
    return linear == nullptr ? 0 : linear->context_bytes;
}

extern "C" void zllm_qnn_linear_destroy(void* handle) { delete static_cast<Linear*>(handle); }
