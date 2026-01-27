#include "napi/native_api.h"
#include "llama.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <hilog/log.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <unistd.h>

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x0000
#define LOG_TAG "MNN_NATIVE"
#define LOGI(...) OH_LOG_Print(LOG_APP, LOG_INFO, LOG_DOMAIN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_DOMAIN, LOG_TAG, __VA_ARGS__)

// 外部 Sherpa 函数声明
extern napi_value InitSherpa(napi_env env, napi_callback_info info);
extern napi_value AcceptWaveform(napi_env env, napi_callback_info info);
extern napi_value ResetSherpa(napi_env env, napi_callback_info info);
extern napi_value GetRecognizedText(napi_env env, napi_callback_info info);
extern napi_value GetQueueSize(napi_env env, napi_callback_info info);

// ==========================================
// LLM 异步化核心变量
// ==========================================
static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;

// 线程安全控制
static std::mutex g_llm_mutex;
static std::string g_llm_input_prompt = "";   // 待处理的问题
static std::string g_llm_output_buffer = "";  // 待取走的答案
static std::atomic<bool> g_llm_running = false;
static std::thread* g_llm_thread = nullptr;

// 🔥 LLM 后台工作线程 🔥
void LlmBackgroundWorker() {
    LOGI("🧵 LLM 后台线程已启动");
    while (g_llm_running) {
        std::string prompt;
        {
            std::lock_guard<std::mutex> lock(g_llm_mutex);
            if (!g_llm_input_prompt.empty()) {
                prompt = g_llm_input_prompt;
                g_llm_input_prompt = ""; // 取走任务
            }
        }

        if (prompt.empty()) {
            usleep(20000); // 没任务就休息 20ms
            continue;
        }

        if (!g_model || !g_ctx) {
            LOGE("❌ 模型未加载，无法推理");
            continue;
        }

        // --- 开始推理 (耗时操作) ---
        LOGI("🤖 LLM 开始思考: %{public}s", prompt.c_str());
        
        // 1. Tokenize
        std::string full_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        const llama_vocab* vocab = llama_model_get_vocab(g_model);
        std::vector<llama_token> tokens(full_prompt.length() + 100);
        int n_tokens = llama_tokenize(vocab, full_prompt.c_str(), full_prompt.length(), tokens.data(), tokens.size(), true, true);
        if (n_tokens < 0) {
            n_tokens = -n_tokens;
            tokens.resize(n_tokens);
            n_tokens = llama_tokenize(vocab, full_prompt.c_str(), full_prompt.length(), tokens.data(), tokens.size(), true, true);
        }
        tokens.resize(n_tokens);

        // 2. Initial Decode
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        llama_decode(g_ctx, batch);

        // 3. Generation Loop
        for (int i = 0; i < 512; i++) { // 最多生成 512 token
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

            // 遇到结束符停止
            if (llama_vocab_is_eog(vocab, next_token)) break;

            // 转为字符串
            char buf[256];
            int n = llama_token_to_piece(vocab, next_token, buf, sizeof(buf), 0, true);
            if (n < 0) {
                 n = -n;
                 llama_token_to_piece(vocab, next_token, buf, n, 0, true);
            }
            buf[n] = '\0';

            // 🔥 将生成的字放入缓冲区，供 JS 拿取 🔥
            {
                std::lock_guard<std::mutex> lock(g_llm_mutex);
                g_llm_output_buffer += std::string(buf);
            }

            // 准备下一次迭代
            batch = llama_batch_get_one(&next_token, 1);
            if (llama_decode(g_ctx, batch) != 0) break;
        }
        
        LOGI("✅ LLM 回复完成");
    }
}

// 1. 加载 LLM (同时启动后台线程)
static napi_value NativeLoad(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    char pathBuf[512];
    size_t strSize;
    napi_get_value_string_utf8(env, args[0], pathBuf, 512, &strSize);

    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_free_model(g_model); g_model = nullptr; }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = false; 

    g_model = llama_model_load_from_file(pathBuf, model_params);
    bool success = (g_model != nullptr);
    
    if (success) {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;
        ctx_params.n_threads = 4;
        ctx_params.n_threads_batch = 4;
        ctx_params.n_batch = 128; 
        g_ctx = llama_new_context_with_model(g_model, ctx_params);
        
        // 🔥 启动后台线程 🔥
        if (!g_llm_running) {
            g_llm_running = true;
            g_llm_thread = new std::thread(LlmBackgroundWorker);
            g_llm_thread->detach();
        }
    }

    napi_value result;
    napi_get_boolean(env, success, &result);
    return result;
}

// 2. 发送问题 (非阻塞，立即返回)
static napi_value NativeChat(napi_env env, napi_callback_info info) {
    size_t argc = 1; 
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    char qBuf[1024];
    size_t strSize;
    napi_get_value_string_utf8(env, args[0], qBuf, 1024, &strSize);
    
    // 只负责把问题放入队列
    {
        std::lock_guard<std::mutex> lock(g_llm_mutex);
        g_llm_input_prompt = std::string(qBuf);
    }

    napi_value result;
    napi_create_string_utf8(env, "OK", NAPI_AUTO_LENGTH, &result);
    return result;
}

// 3. 获取结果 (供 JS 轮询)
static napi_value GetLlmResult(napi_env env, napi_callback_info info) {
    std::string res = "";
    {
        std::lock_guard<std::mutex> lock(g_llm_mutex);
        if (!g_llm_output_buffer.empty()) {
            res = g_llm_output_buffer;
            g_llm_output_buffer = ""; // 取走后清空，实现流式
        }
    }
    napi_value output;
    napi_create_string_utf8(env, res.c_str(), NAPI_AUTO_LENGTH, &output);
    return output;
}

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        // LLM
        {"nativeLoad", nullptr, NativeLoad, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"nativeChat", nullptr, NativeChat, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getLlmResult", nullptr, GetLlmResult, nullptr, nullptr, nullptr, napi_default, nullptr}, // 新增接口
        
        // Sherpa
        {"initSherpa", nullptr, InitSherpa, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"acceptWaveform", nullptr, AcceptWaveform, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"resetSherpa", nullptr, ResetSherpa, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getRecognizedText", nullptr, GetRecognizedText, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getQueueSize", nullptr, GetQueueSize, nullptr, nullptr, nullptr, napi_default, nullptr}
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