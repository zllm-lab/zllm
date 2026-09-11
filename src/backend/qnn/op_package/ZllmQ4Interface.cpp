// QNN HTP 自定义算子包的 ARM prepare 与 DSP 公共入口。
#include "HTP/QnnHtpCommon.h"
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "HTP/core/unique_types.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"

DEFINE_UNIQ_TY()
BEGIN_PKG_OPS_OPTS_LIST()
DECLARE_PKG_OPS_OPTS_LIST(PKG_BlockQ4Gemv)
END_PKG_OPS_OPTS_LIST()

static constexpr auto package_name = THIS_PKG_NAME_STR;
static std::array<const char *, 1> op_names{{"BlockQ4Gemv"}};
static Qnn_ApiVersion_t sdk_api_version = QNN_HTP_API_VERSION_INIT;
static QnnOpPackage_Info_t package_info = QNN_OP_PACKAGE_INFO_INIT;
static QnnOpPackage_GlobalInfrastructure_t global_infrastructure = nullptr;
static bool package_initialized = false;
static QnnLog_Callback_t log_callback = nullptr;
static QnnLog_Level_t max_log_level = static_cast<QnnLog_Level_t>(0);
static bool log_initialized = false;

INIT_PACKAGE_OP_DEF()
INIT_PACKAGE_OPTIMIZATION_DEF()
INIT_PACKAGE_PARAM_ORDER_DEF()
INIT_PKG_CORE_INIT_FUNC()

Qnn_ErrorHandle_t ZllmQ4Init(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
    if (package_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    REGISTER_PACKAGE_PARAM_ORDERS()
    REGISTER_PACKAGE_AXIS_PARAMS()
    REGISTER_PACKAGE_PER_CHANNEL_QUANTIZED_OPS()
    global_infrastructure = infrastructure;
    package_initialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ZllmQ4GetInfo(const QnnOpPackage_Info_t **info) {
    if (!package_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    package_info = QNN_OP_PACKAGE_INFO_INIT;
    package_info.packageName = package_name;
    package_info.operationNames = op_names.data();
    package_info.numOperations = op_names.size();
    package_info.sdkBuildId = QNN_SDK_BUILD_ID;
    package_info.sdkApiVersion = &sdk_api_version;
    *info = &package_info;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ZllmQ4LogInitialize(QnnLog_Callback_t callback, QnnLog_Level_t level) {
    if (log_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    if (!callback) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    if (level < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    log_callback = callback;
    max_log_level = level;
    log_initialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ZllmQ4LogSetLevel(QnnLog_Level_t level) {
    if (level < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    max_log_level = level;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ZllmQ4LogTerminate() {
    if (!log_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    log_callback = nullptr;
    max_log_level = static_cast<QnnLog_Level_t>(0);
    log_initialized = false;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ZllmQ4ValidateOpConfig(Qnn_OpConfig_t config) {
    if (std::string(package_name) != config.v1.packageName ||
        std::string(config.v1.typeName) != "BlockQ4Gemv" || config.v1.numOfParams != 0 ||
        config.v1.numOfInputs != 3 || config.v1.numOfOutputs != 1) {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ZllmQ4Terminate() {
    if (!package_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    global_infrastructure = nullptr;
    package_initialized = false;
    return QNN_SUCCESS;
}

extern "C" Qnn_ErrorHandle_t ZllmQ4InterfaceProvider(QnnOpPackage_Interface_t *interface) {
    if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    interface->interfaceVersion = {1, 4, 0};
    interface->v1_4.init = ZllmQ4Init;
    interface->v1_4.terminate = ZllmQ4Terminate;
    interface->v1_4.getInfo = ZllmQ4GetInfo;
    interface->v1_4.validateOpConfig = ZllmQ4ValidateOpConfig;
    interface->v1_4.createOpImpl = nullptr;
    interface->v1_4.freeOpImpl = nullptr;
    interface->v1_4.logInitialize = ZllmQ4LogInitialize;
    interface->v1_4.logSetLevel = ZllmQ4LogSetLevel;
    interface->v1_4.logTerminate = ZllmQ4LogTerminate;
    return QNN_SUCCESS;
}
