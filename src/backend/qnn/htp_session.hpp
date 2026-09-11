// QNN HTP 会话 bootstrap:统一 dlopen/provider/backend/device、可选错误日志、
// 性能投票与自定义算子包注册,是全部 HTP runner 的公共入口。
//
// HTP 架构(V73/V75/V79...)在本层只体现为算子包 skel 的文件名:HTP 图构建逻辑
// 与量化编码完全架构无关。换新架构设备时,配置换 htp_arch、部署脚本推对应
// Stub/Skel 即可,不需要改任何 runner 代码。
#pragma once

#include "htp_graph.hpp"
#include "qnn_json.hpp"
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpPerfInfrastructure.h>
#include <QnnInterface.h>
#include <System/QnnSystemContext.h>
#include <System/QnnSystemInterface.h>
#include <cstdarg>
#include <cstdio>
#include <dlfcn.h>
#include <functional>
#include <poll.h>
#include <unistd.h>

// 与原 bench runner 源码保持同一短名;全部 HTP 引擎文件经本头获得该别名。
using json = nlohmann::json;

struct HtpSession {
    QNN_INTERFACE_VER_TYPE api{};
    QNN_SYSTEM_INTERFACE_VER_TYPE system{};
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_LogHandle_t logger = nullptr;
    bool has_system = false;

    ~HtpSession() {
        if (logger) api.logFree(logger);
        if (backend) {
            api.deviceFree(device);
            api.backendFree(backend);
        }
    }
    HtpSession() = default;
    // 句柄归本对象独有:默认拷贝会让临时对象的析构释放句柄,留下悬垂的
    // backend/device(make_unique 转发即触发,曾致 ASR 加载 5002)。禁拷贝,
    // 移动时把源句柄置空,由目标独占释放。
    HtpSession(const HtpSession&) = delete;
    HtpSession& operator=(const HtpSession&) = delete;
    HtpSession(HtpSession&& other) noexcept
        : api(other.api), system(other.system), backend(other.backend), device(other.device), logger(other.logger), has_system(other.has_system) {
        other.backend = nullptr;
        other.device = nullptr;
        other.logger = nullptr;
        other.has_system = false;
    }
    HtpSession& operator=(HtpSession&& other) noexcept {
        if (this != &other) {
            if (logger) api.logFree(logger);
            if (backend) {
                api.deviceFree(device);
                api.backendFree(backend);
            }
            api = other.api;
            system = other.system;
            backend = other.backend;
            device = other.device;
            logger = other.logger;
            has_system = other.has_system;
            other.backend = nullptr;
            other.device = nullptr;
            other.logger = nullptr;
            other.has_system = false;
        }
        return *this;
    }
};

// need_system=true 时同时加载 libQnnSystem.so,context binary 元数据读取需要它。
// 配置键(全部可选):
//   qnn_error_log       打开 QNN 错误日志到 stderr
//   performance_mode    HTP performance infrastructure 模式(部分厂商库上崩溃,默认关闭)
//   performance_vote    DCVS 性能投票;performance_turbo/performance_max 控制电压角
//   register_q4_package 注册 ZllmQ4 自定义算子包;htp_arch 决定 HTP skel 文件名
//   htp_arch            默认 "v73";V75/V79 设备部署对应 skel 后改为 "v75"/"v79"
//   q4_package_dir      算子包目录,默认 "/data/local/tmp"
//   q4_package_arm_path / q4_package_htp_path  完整路径覆盖(调试用)
inline HtpSession htp_session_open(const json& cfg, bool need_system = false) {
    HtpSession session;
    void* lib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) throw std::runtime_error(dlerror());
    using Providers = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
    const QnnInterface_t** ps = nullptr;
    uint32_t count = 0;
    check(((Providers)dlsym(lib, "QnnInterface_getProviders"))(&ps, &count), "providers");
    if (!count) throw std::runtime_error("HTP provider 为空");
    session.api = ps[0]->QNN_INTERFACE_VER_NAME;
    if (need_system) {
        void* system_lib = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
        if (!system_lib) throw std::runtime_error(dlerror());
        using SystemProviders = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);
        const QnnSystemInterface_t** sps = nullptr;
        check(((SystemProviders)dlsym(system_lib, "QnnSystemInterface_getProviders"))(&sps, &count), "system providers");
        if (!count) throw std::runtime_error("system provider 为空");
        session.system = sps[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
        session.has_system = true;
    }
    if (cfg.value("qnn_error_log", false)) {
        check(session.api.logCreate([](const char* format, QnnLog_Level_t, uint64_t, va_list args) { vfprintf(stderr, format, args); fputc('\n', stderr); }, QNN_LOG_LEVEL_ERROR, &session.logger), "logger");
    }
    check(session.api.backendCreate(session.logger, nullptr, &session.backend), "backend");
    check(session.api.deviceCreate(nullptr, nullptr, &session.device), "device");
    if (cfg.value("performance_mode", false)) {
        QnnDevice_Infrastructure_t raw = nullptr;
        check(session.api.deviceGetInfrastructure(&raw), "device infrastructure");
        auto* infra = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(raw);
        if (infra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) throw std::runtime_error("HTP performance infrastructure 类型不匹配");
        uint32_t power_id = 0;
        check(infra->perfInfra.createPowerConfigId(0, 0, &power_id), "power config id");
        QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        dcvs.dcvsV3Config = {power_id, 1, 0, QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE, 1, 1, 1, 1, 0, DCVS_VOLTAGE_CORNER_DISABLE, DCVS_VOLTAGE_CORNER_DISABLE, DCVS_VOLTAGE_CORNER_DISABLE, 0, DCVS_VOLTAGE_CORNER_DISABLE, DCVS_VOLTAGE_CORNER_DISABLE, DCVS_VOLTAGE_CORNER_DISABLE};
        const QnnHtpPerfInfrastructure_PowerConfig_t* power_configs[] = {&dcvs, nullptr};
        check(infra->perfInfra.setPowerConfig(power_id, power_configs), "set performance mode");
        printf("HTP performance mode enabled\n");
    }
    if (cfg.value("performance_vote", false)) {
        QnnDevice_Infrastructure_t raw = nullptr;
        check(session.api.deviceGetInfrastructure(&raw), "device infrastructure");
        auto* infra = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(raw);
        if (infra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) throw std::runtime_error("HTP performance infrastructure 类型不匹配");
        uint32_t id = 0;
        check(infra->perfInfra.createPowerConfigId(0, 0, &id), "power config id");
        QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        dcvs.dcvsV3Config.contextId = id;
        dcvs.dcvsV3Config.setDcvsEnable = 1;
        dcvs.dcvsV3Config.dcvsEnable = 0;
        dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        if (cfg.value("performance_turbo", false)) {
            auto& v = dcvs.dcvsV3Config;
            auto corner = cfg.value("performance_max", false) ? DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER : DCVS_VOLTAGE_VCORNER_TURBO;
            v.setSleepLatency = 1;
            v.sleepLatency = 40;
            v.setBusParams = 1;
            v.busVoltageCornerMin = v.busVoltageCornerTarget = v.busVoltageCornerMax = corner;
            v.setCoreParams = 1;
            v.coreVoltageCornerMin = v.coreVoltageCornerTarget = v.coreVoltageCornerMax = corner;
        }
        QnnHtpPerfInfrastructure_PowerConfig_t polling = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
        polling.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
        polling.rpcPollingTimeConfig = 9999;
        const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&dcvs, &polling, nullptr};
        check(infra->perfInfra.setPowerConfig(id, configs), "performance vote");
        printf("HTP performance vote enabled\n");
    }
    if (cfg.value("register_q4_package", false)) {
        std::string dir = cfg.value("q4_package_dir", std::string("/data/local/tmp"));
        std::string arch = cfg.value("htp_arch", std::string("v73"));
        std::string arm_path = cfg.value("q4_package_arm_path", dir + "/libZllmQ4-arm.so");
        std::string htp_path = cfg.value("q4_package_htp_path", dir + "/libZllmQ4-" + arch + ".so");
        check(session.api.backendRegisterOpPackage(session.backend, arm_path.c_str(), "ZllmQ4InterfaceProvider", "CPU"), "register ARM Q4 package");
        check(session.api.backendRegisterOpPackage(session.backend, htp_path.c_str(), "ZllmQ4InterfaceProvider", "HTP"), ("register HTP Q4 package " + htp_path).c_str());
        printf("ZllmQ4 op package registered arch=%s\n", arch.c_str());
    }
    return session;
}

// context binary 元数据:graph 名与输入输出张量,加载缓存图时使用。
struct GraphInfo {
    const char* name;
    Qnn_Tensor_t* inputs;
    uint32_t input_count;
    Qnn_Tensor_t* outputs;
    uint32_t output_count;
};
inline GraphInfo graph_info(const QnnSystemContext_BinaryInfo_t* binary) {
    QnnSystemContext_GraphInfo_t* graph = nullptr;
    switch (binary->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: graph = binary->contextBinaryInfoV1.graphs; break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: graph = binary->contextBinaryInfoV2.graphs; break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: graph = binary->contextBinaryInfoV3.graphs; break;
        default: throw std::runtime_error("不支持的 context binary 版本");
    }
    switch (graph->version) {
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1: {
            auto& v = graph->graphInfoV1;
            return {v.graphName, v.graphInputs, v.numGraphInputs, v.graphOutputs, v.numGraphOutputs};
        }
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2: {
            auto& v = graph->graphInfoV2;
            return {v.graphName, v.graphInputs, v.numGraphInputs, v.graphOutputs, v.numGraphOutputs};
        }
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3: {
            auto& v = graph->graphInfoV3;
            return {v.graphName, v.graphInputs, v.numGraphInputs, v.graphOutputs, v.numGraphOutputs};
        }
        default: throw std::runtime_error("不支持的 graph info 版本");
    }
}

// 常驻模式从 stdin 逐行读取请求配置路径;空闲时轮询,SIGTERM 置位后能及时退出。
// 返回 1 表示读到一行,0 表示超时,-1 表示对端关闭或出错。
inline int htp_read_line_poll(int fd, std::string& buffer, std::string& line, int timeout_ms) {
    while (buffer.find('\n') == std::string::npos) {
        struct pollfd waiter{fd, POLLIN, 0};
        int ready = poll(&waiter, 1, timeout_ms);
        if (ready < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (ready == 0) return 0;
        char chunk[4096];
        ssize_t got = read(fd, chunk, sizeof(chunk));
        if (got < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (got == 0) return -1;
        buffer.append(chunk, (size_t)got);
    }
    auto newline = buffer.find('\n');
    line = buffer.substr(0, newline);
    buffer.erase(0, newline + 1);
    while (!line.empty() && line.back() == '\r') line.pop_back();
    return 1;
}
