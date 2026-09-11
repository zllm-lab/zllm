// QNN HTP 图构建公共层:Value 量化缓冲与 Graph 逐算子建图封装。
// 模型无关、HTP arch 无关:量化编码(per-channel/低比特/块编码)只描述数据布局,
// 不含任何 V73/V75 等架构分支,新架构设备无需修改本文件。
#pragma once

#include <QnnInterface.h>
#include <QnnOpDef.h>
#include <HTP/QnnHtpGraph.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

inline void check(Qnn_ErrorHandle_t rc, const char* s) {
    if (rc) throw std::runtime_error(std::string(s) + " status=" + std::to_string(rc));
}
template <class T>
std::vector<T> read(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("open " + p);
    size_t n = f.tellg();
    if (n % sizeof(T)) throw std::runtime_error("size " + p);
    std::vector<T> v(n / sizeof(T));
    f.seekg(0);
    f.read((char*)v.data(), n);
    if (!f) throw std::runtime_error("read " + p);
    return v;
}
template <class T>
void write(const std::string& p, const std::vector<T>& v) {
    std::ofstream f(p, std::ios::binary);
    f.write((const char*)v.data(), v.size() * sizeof(T));
    if (!f) throw std::runtime_error("write " + p);
}
struct Value {
    std::vector<uint16_t> data;
    uint32_t rows, cols;
    float step;
    std::vector<float> row_steps;
};
inline uint32_t argmax(const Value& value, const std::vector<unsigned>& excluded = {}) {
    std::vector<uint8_t> mask(value.data.size());
    for (unsigned id : excluded)
        if (id < mask.size()) mask[id] = 1;
    uint32_t best = 0;
    while (best < mask.size() && mask[best]) best++;
    if (best == mask.size()) throw std::runtime_error("argmax 没有可选 token");
    for (uint32_t i = best + 1; i < value.data.size(); i++)
        if (!mask[i] && value.data[i] > value.data[best]) best = i;
    return best;
}
static Qnn_QuantizeParams_t quant(float s, int z = -32768) {
    Qnn_QuantizeParams_t q = QNN_QUANTIZE_PARAMS_INIT;
    q.encodingDefinition = QNN_DEFINITION_DEFINED;
    q.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    q.scaleOffsetEncoding = {s, z};
    return q;
}
struct Graph {
    QNN_INTERFACE_VER_TYPE& a;
    Qnn_ContextHandle_t ctx = nullptr;
    Qnn_GraphHandle_t g = nullptr;
    bool finalized = false;
    std::deque<std::string> names;
    std::deque<std::vector<uint32_t>> dims, block_sizes;
    std::deque<std::vector<uint8_t>> buffers, block_multipliers;
    std::deque<std::vector<float>> axis_scales;
    std::deque<std::vector<Qnn_ScaleOffset_t>> block_scales;
    std::deque<Qnn_BlockEncoding_t> block_encodings;
    std::vector<Qnn_Tensor_t> ins, outs;
    std::map<Value*, Qnn_Tensor_t> input_tensors;
    Graph(QNN_INTERFACE_VER_TYPE& api, Qnn_BackendHandle_t b, Qnn_DeviceHandle_t d) : a(api) {
        check(a.contextCreate(b, d, nullptr, &ctx), "context");
        QnnHtpGraph_CustomConfig_t custom = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        custom.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        custom.optimizationOption = {QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG, 1.0f};
        QnnHtpGraph_CustomConfig_t packing = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        packing.option = QNN_HTP_GRAPH_CONFIG_OPTION_WEIGHTS_PACKING;
        packing.weightsPacking = true;
        QnnGraph_Config_t config = QNN_GRAPH_CONFIG_INIT, packing_config = QNN_GRAPH_CONFIG_INIT;
        config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        config.customConfig = &custom;
        packing_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        packing_config.customConfig = &packing;
        const QnnGraph_Config_t* configs[] = {&config, &packing_config, nullptr};
        check(a.graphCreate(ctx, "stage", configs, &g), "graph");
    }
    ~Graph() {
        if (ctx) a.contextFree(ctx, nullptr);
    }
    const char* name() {
        names.push_back("v" + std::to_string(names.size()));
        return names.back().c_str();
    }
    Qnn_Tensor_t tensor(std::vector<uint32_t> shape, float step, Qnn_TensorType_t type = QNN_TENSOR_TYPE_NATIVE, void* data = nullptr, uint32_t bytes = 0, Qnn_DataType_t dt = QNN_DATATYPE_UFIXED_POINT_16, int offset = -32768) {
        dims.push_back(shape);
        Qnn_Tensor_t t = QNN_TENSOR_INIT;
        t.v1.name = name();
        t.v1.type = type;
        t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        t.v1.dataType = dt;
        t.v1.rank = shape.size();
        t.v1.dimensions = dims.back().data();
        t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        t.v1.clientBuf = {data, bytes};
        if (step > 0) t.v1.quantizeParams = quant(step, offset);
        check(a.tensorCreateGraphTensor(g, &t), t.v1.name);
        return t;
    }
    Qnn_Tensor_t row_tensor(std::vector<uint32_t> shape, const std::vector<float>& steps, Qnn_TensorType_t type = QNN_TENSOR_TYPE_NATIVE) {
        if (shape.empty() || steps.size() != shape[0]) throw std::runtime_error("逐行量化 scale 数量不匹配");
        dims.push_back(shape);
        block_scales.emplace_back(steps.size());
        for (size_t i = 0; i < steps.size(); i++) block_scales.back()[i] = {steps[i], 0};
        Qnn_QuantizeParams_t q = QNN_QUANTIZE_PARAMS_INIT;
        q.encodingDefinition = QNN_DEFINITION_DEFINED;
        q.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
        q.axisScaleOffsetEncoding.axis = 0;
        q.axisScaleOffsetEncoding.numScaleOffsets = steps.size();
        q.axisScaleOffsetEncoding.scaleOffset = block_scales.back().data();
        Qnn_Tensor_t t = QNN_TENSOR_INIT;
        t.v1.name = name();
        t.v1.type = type;
        t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        t.v1.dataType = QNN_DATATYPE_SFIXED_POINT_16;
        t.v1.rank = shape.size();
        t.v1.dimensions = dims.back().data();
        t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        check(a.tensorCreateGraphTensor(g, &t), t.v1.name);
        return t;
    }
    Qnn_Tensor_t q4(std::vector<uint32_t> shape, void* data, uint32_t bytes, const std::vector<float>& scales, uint32_t block) {
        uint32_t blocks = shape[0] / block, channels = shape[1];
        if (scales.size() != (size_t)blocks * channels) throw std::runtime_error("Q4 block scale 数量不匹配");
        dims.push_back(shape);
        block_sizes.push_back({block, 1});
        block_scales.emplace_back(scales.size());
        for (size_t i = 0; i < scales.size(); i++) block_scales.back()[i] = {scales[i], 0};
        Qnn_BlockEncoding_t encoding = QNN_BLOCK_ENCODING_INIT;
        encoding.blockSize = block_sizes.back().data();
        encoding.scaleOffset = block_scales.back().data();
        block_encodings.push_back(encoding);
        Qnn_QuantizeParams_t q = QNN_QUANTIZE_PARAMS_INIT;
        q.encodingDefinition = QNN_DEFINITION_DEFINED;
        q.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BLOCK;
        q.blockEncoding = block_encodings.back();
        Qnn_Tensor_t t = QNN_TENSOR_INIT;
        t.v1.name = name();
        t.v1.type = QNN_TENSOR_TYPE_STATIC;
        t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        t.v1.dataType = QNN_DATATYPE_SFIXED_POINT_8;
        t.v1.rank = shape.size();
        t.v1.dimensions = dims.back().data();
        t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        t.v1.clientBuf = {data, bytes};
        t.v1.quantizeParams = q;
        check(a.tensorCreateGraphTensor(g, &t), t.v1.name);
        return t;
    }
    Qnn_Tensor_t lowbit(std::vector<uint32_t> shape, void* data, uint32_t bytes, std::vector<float> scales, uint32_t bits) {
        dims.push_back(shape);
        axis_scales.push_back(std::move(scales));
        Qnn_QuantizeParams_t q = QNN_QUANTIZE_PARAMS_INIT;
        q.encodingDefinition = QNN_DEFINITION_DEFINED;
        if (bits == 8) {
            block_scales.emplace_back(axis_scales.back().size());
            for (size_t i = 0; i < axis_scales.back().size(); i++) block_scales.back()[i] = {axis_scales.back()[i], 0};
            q.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
            q.axisScaleOffsetEncoding.axis = 1;
            q.axisScaleOffsetEncoding.numScaleOffsets = shape[1];
            q.axisScaleOffsetEncoding.scaleOffset = block_scales.back().data();
        } else {
            q.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
            q.bwAxisScaleOffsetEncoding.bitwidth = bits;
            q.bwAxisScaleOffsetEncoding.axis = 1;
            q.bwAxisScaleOffsetEncoding.numElements = shape[1];
            q.bwAxisScaleOffsetEncoding.scales = axis_scales.back().data();
            q.bwAxisScaleOffsetEncoding.offsets = nullptr;
        }
        Qnn_Tensor_t t = QNN_TENSOR_INIT;
        t.v1.name = name();
        t.v1.type = QNN_TENSOR_TYPE_STATIC;
        t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        t.v1.dataType = bits == 16 ? QNN_DATATYPE_SFIXED_POINT_16 : QNN_DATATYPE_SFIXED_POINT_8;
        t.v1.rank = shape.size();
        t.v1.dimensions = dims.back().data();
        t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        t.v1.clientBuf = {data, bytes};
        t.v1.quantizeParams = q;
        check(a.tensorCreateGraphTensor(g, &t), t.v1.name);
        return t;
    }
    Qnn_Tensor_t input(Value& v) {
        if (input_tensors.count(&v)) return input_tensors.at(&v);
        auto t = v.row_steps.empty() ? tensor({v.rows, v.cols}, v.step, QNN_TENSOR_TYPE_APP_WRITE) : row_tensor({v.rows, v.cols}, v.row_steps, QNN_TENSOR_TYPE_APP_WRITE);
        t.v1.clientBuf = {v.data.data(), (uint32_t)v.data.size() * 2};
        ins.push_back(t);
        input_tensors[&v] = t;
        return t;
    }
    Qnn_Tensor_t output(Value& v) {
        auto t = v.row_steps.empty() ? tensor({v.rows, v.cols}, v.step, QNN_TENSOR_TYPE_APP_READ) : row_tensor({v.rows, v.cols}, v.row_steps, QNN_TENSOR_TYPE_APP_READ);
        t.v1.clientBuf = {v.data.data(), (uint32_t)v.data.size() * 2};
        outs.push_back(t);
        return t;
    }
    Qnn_Param_t scalar(const char* n, uint32_t v, Qnn_DataType_t dt = QNN_DATATYPE_UINT_32) {
        Qnn_Param_t p = QNN_PARAM_INIT;
        p.name = n;
        p.paramType = QNN_PARAMTYPE_SCALAR;
        p.scalarParam.dataType = dt;
        p.scalarParam.uint32Value = v;
        return p;
    }
    Qnn_Param_t array(const char* n, std::vector<uint32_t> v) {
        buffers.emplace_back(v.size() * 4);
        memcpy(buffers.back().data(), v.data(), v.size() * 4);
        auto t = tensor({(uint32_t)v.size()}, 0, QNN_TENSOR_TYPE_STATIC, buffers.back().data(), v.size() * 4, QNN_DATATYPE_UINT_32);
        Qnn_Param_t p = QNN_PARAM_INIT;
        p.name = n;
        p.paramType = QNN_PARAMTYPE_TENSOR;
        p.tensorParam = t;
        return p;
    }
    void op(const char* type, std::vector<Qnn_Tensor_t> in, Qnn_Tensor_t out, std::vector<Qnn_Param_t> p = {}) {
        Qnn_OpConfig_t o = QNN_OPCONFIG_INIT;
        o.v1.name = name();
        o.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        o.v1.typeName = type;
        o.v1.numOfInputs = in.size();
        o.v1.inputTensors = in.data();
        o.v1.numOfOutputs = 1;
        o.v1.outputTensors = &out;
        o.v1.numOfParams = p.size();
        o.v1.params = p.data();
        check(a.graphAddNode(g, o), type);
    }
    void custom_op(const char* package, const char* type, std::vector<Qnn_Tensor_t> in, Qnn_Tensor_t out) {
        Qnn_OpConfig_t o = QNN_OPCONFIG_INIT;
        o.v1.name = name();
        o.v1.packageName = package;
        o.v1.typeName = type;
        o.v1.numOfInputs = in.size();
        o.v1.inputTensors = in.data();
        o.v1.numOfOutputs = 1;
        o.v1.outputTensors = &out;
        check(a.graphAddNode(g, o), type);
    }
    // 常量编码在图准备阶段完成，数值不依赖手机上的 activation。
    Qnn_Tensor_t constant(std::vector<float> v, std::vector<uint32_t> shape, float step = 0) {
        float mx = 0;
        bool positive = true;
        for (float x : v) {
            mx = std::max(mx, std::abs(x));
            positive = positive && x >= 0;
        }
        int zero = positive ? 0 : 32768;
        if (step == 0) step = std::max(mx / (positive ? 65535 : 32767), 1e-9f);
        buffers.emplace_back(v.size() * 2);
        for (size_t i = 0; i < v.size(); i++) {
            uint16_t q = std::clamp((int)std::round(v[i] / step) + zero, 0, 65535);
            memcpy(buffers.back().data() + i * 2, &q, 2);
        }
        return tensor(shape, step, QNN_TENSOR_TYPE_STATIC, buffers.back().data(), v.size() * 2, QNN_DATATYPE_UFIXED_POINT_16, -zero);
    }
    Qnn_Tensor_t constant(float x) { return constant({x}, {1, 1}); }
    Qnn_Tensor_t reshape(Qnn_Tensor_t in, std::vector<uint32_t> d) {
        auto encoding = in.v1.quantizeParams.scaleOffsetEncoding;
        auto t = tensor(d, encoding.scale, QNN_TENSOR_TYPE_NATIVE, nullptr, 0, in.v1.dataType, encoding.offset);
        op(QNN_OP_RESHAPE, {in}, t);
        return t;
    }
    Qnn_Tensor_t slice(Qnn_Tensor_t in, uint32_t start, uint32_t count, uint32_t axis = 1) {
        if (axis >= in.v1.rank || start > in.v1.dimensions[axis] || !count || count > in.v1.dimensions[axis] - start) throw std::runtime_error("连续切片范围无效");
        // 连续区间使用 StridedSlice；Gather 在大 KV 的列切片上会变成昂贵的逐项索引。
        std::vector<int32_t> ranges(in.v1.rank * 3);
        for (unsigned d = 0; d < in.v1.rank; d++) {
            ranges[d * 3] = d == axis ? start : 0;
            ranges[d * 3 + 1] = d == axis ? start + count : in.v1.dimensions[d];
            ranges[d * 3 + 2] = 1;
        }
        buffers.emplace_back(ranges.size() * 4);
        memcpy(buffers.back().data(), ranges.data(), buffers.back().size());
        Qnn_Param_t param = QNN_PARAM_INIT;
        param.name = QNN_OP_STRIDED_SLICE_PARAM_RANGES;
        param.paramType = QNN_PARAMTYPE_TENSOR;
        param.tensorParam = tensor({in.v1.rank, 3}, 0, QNN_TENSOR_TYPE_STATIC, buffers.back().data(), buffers.back().size(), QNN_DATATYPE_INT_32);
        auto shape = std::vector<uint32_t>(in.v1.dimensions, in.v1.dimensions + in.v1.rank);
        shape[axis] = count;
        auto encoding = in.v1.quantizeParams.scaleOffsetEncoding;
        auto out = tensor(shape, encoding.scale, QNN_TENSOR_TYPE_NATIVE, nullptr, 0, in.v1.dataType, encoding.offset);
        op(QNN_OP_STRIDED_SLICE, {in}, out, {param});
        return out;
    }
    Qnn_Tensor_t binary(const char* opname, Qnn_Tensor_t x, Qnn_Tensor_t y, std::vector<uint32_t> shape, float step) {
        auto t = tensor(shape, step);
        op(opname, {x, y}, t);
        return t;
    }
    Qnn_Tensor_t binary_rows(const char* opname, Qnn_Tensor_t x, Qnn_Tensor_t y, std::vector<uint32_t> shape, const std::vector<float>& steps) {
        auto t = row_tensor(shape, steps);
        op(opname, {x, y}, t);
        return t;
    }
    void finalize() {
        if (!finalized) {
            check(a.graphFinalize(g, nullptr, nullptr), "finalize");
            finalized = true;
        }
    }
    void save(const std::string& path) {
        finalize();
        uint64_t size = 0, written = 0;
        check(a.contextGetBinarySize(ctx, &size), "context binary size");
        std::vector<uint8_t> bytes(size);
        check(a.contextGetBinary(ctx, bytes.data(), bytes.size(), &written), "context binary");
        bytes.resize(written);
        write(path, bytes);
    }
    void release_build_buffers() {
        buffers.clear();
        block_sizes.clear();
        block_multipliers.clear();
        axis_scales.clear();
        block_scales.clear();
        block_encodings.clear();
    }
    void execute(uint32_t repeats = 0) {
        finalize();
        check(a.graphExecute(g, ins.data(), ins.size(), outs.data(), outs.size(), nullptr, nullptr), "execute");
        if (repeats) {
            for (unsigned i = 0; i < 5; i++) check(a.graphExecute(g, ins.data(), ins.size(), outs.data(), outs.size(), nullptr, nullptr), "warmup");
            auto begin = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < repeats; i++) check(a.graphExecute(g, ins.data(), ins.size(), outs.data(), outs.size(), nullptr, nullptr), "repeat");
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / repeats;
            printf("HOT repeats=%u mean_ms=%.6f\n", repeats, ms);
        }
    }
};
