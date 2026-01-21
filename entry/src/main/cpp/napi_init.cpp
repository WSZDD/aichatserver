#include "napi/native_api.h"
#include "llama.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <hilog/log.h>

// 定义日志标签和 Domain
#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x0000
#define LOG_TAG "MNN_NATIVE"

// ✅ 修改日志宏，使用 OH_LOG_Print
#define LOGI(...) OH_LOG_Print(LOG_APP, LOG_INFO, LOG_DOMAIN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_DOMAIN, LOG_TAG, __VA_ARGS__)

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;

// 1. 加载模型
static napi_value NativeLoad(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    char pathBuf[512];
    size_t strSize;
    napi_get_value_string_utf8(env, args[0], pathBuf, 512, &strSize);

    LOGI("🚀 Loading model: %s", pathBuf);

    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_free_model(g_model); g_model = nullptr; }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = false; 

    g_model = llama_model_load_from_file(pathBuf, model_params);
    
    bool success = (g_model != nullptr);
    if (success) {
        llama_context_params ctx_params = llama_context_default_params();
        // 针对 RK3588 32位的优化参数
        ctx_params.n_ctx = 2048;
        ctx_params.n_threads = 4;
        ctx_params.n_threads_batch = 4;
        ctx_params.n_batch = 128; 
        
        g_ctx = llama_new_context_with_model(g_model, ctx_params);
        if (!g_ctx) success = false;
        else LOGI("✅ Context initialized");
    } else {
        LOGE("❌ Load failed");
    }

    napi_value result;
    napi_get_boolean(env, success, &result);
    return result;
}

// 2. 流式对话
static napi_value NativeChat(napi_env env, napi_callback_info info) {
    size_t argc = 2; 
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    char qBuf[1024];
    size_t strSize;
    napi_get_value_string_utf8(env, args[0], qBuf, 1024, &strSize);
    
    napi_value callbackFunc = args[1];

    if (!g_model || !g_ctx) return nullptr;

    // Prompt 模板
    std::string prompt = "<|im_start|>user\n" + std::string(qBuf) + "<|im_end|>\n<|im_start|>assistant\n";

    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    std::vector<llama_token> tokens(prompt.length() + 100);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
        tokens.resize(n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    // Prefill
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_decode(g_ctx, batch);

    // 生成循环
    for (int i = 0; i < 256; i++) { 
        auto * logits = llama_get_logits_ith(g_ctx, batch.n_tokens - 1);
        int n_vocab = llama_vocab_n_tokens(vocab);
        
        llama_token next_token = 0;
        float max_p = -1e9;
        for (int j = 0; j < n_vocab; j++) {
            if (logits[j] > max_p) {
                max_p = logits[j];
                next_token = j;
            }
        }

        if (llama_vocab_is_eog(vocab, next_token)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, next_token, buf, sizeof(buf), 0, true);
        if (n < 0) {
             n = -n;
             llama_token_to_piece(vocab, next_token, buf, n, 0, true);
        }
        buf[n] = '\0';

        // 回调 JS
        napi_value jsToken;
        napi_create_string_utf8(env, buf, NAPI_AUTO_LENGTH, &jsToken);
        napi_value undefined;
        napi_get_undefined(env, &undefined);
        napi_call_function(env, undefined, callbackFunc, 1, &jsToken, nullptr);

        batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(g_ctx, batch) != 0) break;
    }

    napi_value result;
    napi_create_string_utf8(env, "DONE", NAPI_AUTO_LENGTH, &result);
    return result;
}

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"nativeLoad", nullptr, NativeLoad, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"nativeChat", nullptr, NativeChat, nullptr, nullptr, nullptr, napi_default, nullptr}
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module demoModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "mnnllm",
    .nm_priv = ((void*)0),
    .reserved = { 0 },
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void) {
    napi_module_register(&demoModule);
}