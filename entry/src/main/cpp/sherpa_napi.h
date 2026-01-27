// entry/src/main/cpp/sherpa_napi.h
#ifndef SHERPA_NAPI_H
#define SHERPA_NAPI_H

#include "napi/native_api.h"

// 声明 Sherpa 的三个核心函数
napi_value InitSherpa(napi_env env, napi_callback_info info);
napi_value AcceptWaveform(napi_env env, napi_callback_info info);
napi_value ResetSherpa(napi_env env, napi_callback_info info);

#endif // SHERPA_NAPI_H