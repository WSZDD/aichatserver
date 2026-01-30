#include "napi/native_api.h"
#include "llama.h"
#include "tts_manager.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <hilog/log.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <unistd.h>
#include <iostream>

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

static std::mutex g_llm_mutex;
static std::string g_llm_input_prompt = "";
static std::string g_llm_output_buffer = "";
static std::atomic<bool> g_llm_running = false;
static std::thread* g_llm_thread = nullptr;

// 🔥 TTS 专用分句缓冲区 🔥
static std::string g_sentence_accumulator = "";

// 分句结果结构体
struct SplitInfo {
    bool found;      // 是否找到标点
    size_t startPos; // 标点开始的位置
    size_t length;   // 标点本身的长度(中文3字节，英文1字节)
};

// 🔥 修复后的标点查找函数：精确匹配字符串，绝不切断 UTF-8 🔥
SplitInfo FindFirstPunctuation(const std::string& text) {
    // 定义标点列表 (按优先级排序，长的在前)
    static const std::vector<std::string> delims = {
        "，", "。", "？", "！", "；", "：", "\n", // 中文标点
        ",", ".", "?", "!", ";", ":"             // 英文标点
    };

    size_t bestPos = std::string::npos;
    size_t bestLen = 0;

    for (const auto& delim : delims) {
        size_t pos = text.find(delim); // 使用 find 而不是 find_last_of
        if (pos != std::string::npos) {
            // 我们希望找到最靠前的标点，以便尽快朗读
            if (bestPos == std::string::npos || pos < bestPos) {
                bestPos = pos;
                bestLen = delim.length();
            }
        }
    }

    if (bestPos != std::string::npos) {
        return {true, bestPos, bestLen};
    }
    return {false, 0, 0};
}

// 🔥 LLM 后台工作线程 🔥
void LlmBackgroundWorker() {
    LOGI("🧵 LLM 后台线程已启动");
    while (g_llm_running) {
        std::string prompt;
        {
            std::lock_guard<std::mutex> lock(g_llm_mutex);
            if (!g_llm_input_prompt.empty()) {
                prompt = g_llm_input_prompt;
                g_llm_input_prompt = "";
                // 新任务开始：彻底清空 TTS 缓冲区
                g_sentence_accumulator = ""; 
            }
        }

        if (prompt.empty()) {
            usleep(20000); 
            continue;
        }

        if (!g_model || !g_ctx) {
            LOGE("❌ 模型未加载");
            continue;
        }

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

        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(g_ctx, batch) != 0) {
            LOGE("❌ Llama decode failed");
            continue;
        }

        // 3. Generation Loop
        for (int i = 0; i < 512; i++) {
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
            std::string piece(buf);

            // 🔥 核心修改：安全的循环分句逻辑 🔥
            {
                std::lock_guard<std::mutex> lock(g_llm_mutex);
                
                g_llm_output_buffer += piece; // 给界面显示
                g_sentence_accumulator += piece; // 给 TTS 缓冲

                // 循环检查：如果缓冲区里有完整的句子（可能不止一句），就切下来发送
                while (true) {
                    SplitInfo info = FindFirstPunctuation(g_sentence_accumulator);
                    
                    if (info.found) {
                        // 计算截取长度：标点位置 + 标点长度
                        size_t cutLength = info.startPos + info.length;
                        
                        std::string sentence = g_sentence_accumulator.substr(0, cutLength);
                        
                        // 发送这一句给 TTS
                        if (!sentence.empty()) {
                            LOGI("🗣️ 完整分句 TTS: %{public}s", sentence.c_str());
                            TtsManager::Instance().PushText(sentence);
                        }
                        
                        // 从缓冲区移除这一句，保留剩下的
                        g_sentence_accumulator = g_sentence_accumulator.substr(cutLength);
                    } else {
                        // 没找到标点，但如果太长了 (超过60字节，约20汉字)，强制切断防止卡顿
                        if (g_sentence_accumulator.length() > 60) {
                             LOGI("🗣️ 长度强制 TTS: %{public}s", g_sentence_accumulator.c_str());
                             TtsManager::Instance().PushText(g_sentence_accumulator);
                             g_sentence_accumulator = "";
                        }
                        break; // 退出循环，等待下一个 Token
                    }
                }
            }

            batch = llama_batch_get_one(&next_token, 1);
            if (llama_decode(g_ctx, batch) != 0) break;
        }
        
        // 4. 收尾：把剩下的文本也发出去
        {
            std::lock_guard<std::mutex> lock(g_llm_mutex);
            if (!g_sentence_accumulator.empty()) {
                 LOGI("🗣️ 剩余文本 TTS: %{public}s", g_sentence_accumulator.c_str());
                 TtsManager::Instance().PushText(g_sentence_accumulator);
                 g_sentence_accumulator = "";
            }
        }
        
        LOGI("✅ LLM 回复完成");
    }
}

// 1. 加载 LLM
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
        ctx_params.n_threads = 2; 
        ctx_params.n_threads_batch = 2;
        ctx_params.n_batch = 128; 
        g_ctx = llama_new_context_with_model(g_model, ctx_params);
        
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

// 2. 发送问题
static napi_value NativeChat(napi_env env, napi_callback_info info) {
    size_t argc = 1; 
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    char qBuf[1024];
    size_t strSize;
    napi_get_value_string_utf8(env, args[0], qBuf, 1024, &strSize);
    
    // 停止 TTS 播放
    TtsManager::Instance().Stop();

    {
        std::lock_guard<std::mutex> lock(g_llm_mutex);
        g_llm_input_prompt = std::string(qBuf);
        g_llm_output_buffer = ""; 
        g_sentence_accumulator = ""; // 清空缓冲区
    }

    napi_value result;
    napi_create_string_utf8(env, "OK", NAPI_AUTO_LENGTH, &result);
    return result;
}

// 3. 获取 LLM 文本
static napi_value GetLlmResult(napi_env env, napi_callback_info info) {
    std::string res = "";
    {
        std::lock_guard<std::mutex> lock(g_llm_mutex);
        if (!g_llm_output_buffer.empty()) {
            res = g_llm_output_buffer;
            g_llm_output_buffer = ""; 
        }
    }
    napi_value output;
    napi_create_string_utf8(env, res.c_str(), NAPI_AUTO_LENGTH, &output);
    return output;
}

// 4. 初始化 TTS
static napi_value InitTts(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    char pathBuf[512];
    size_t strSize;
    napi_get_value_string_utf8(env, args[0], pathBuf, 512, &strSize);

    bool ret = TtsManager::Instance().Init(std::string(pathBuf));
    
    napi_value result;
    napi_get_boolean(env, ret, &result);
    return result;
}

// 5. 获取 TTS 音频
static napi_value GetTtsAudio(napi_env env, napi_callback_info info) {
    std::vector<int16_t> pcm = TtsManager::Instance().PopAudio();
    if (pcm.empty()) return nullptr;

    void* data;
    napi_value arraybuffer;
    size_t byteLength = pcm.size() * sizeof(int16_t);
    napi_create_arraybuffer(env, byteLength, &data, &arraybuffer);
    memcpy(data, pcm.data(), byteLength);
    return arraybuffer;
}

// 6. 停止 TTS
static napi_value StopTts(napi_env env, napi_callback_info info) {
    TtsManager::Instance().Stop();
    {
        std::lock_guard<std::mutex> lock(g_llm_mutex);
        g_llm_input_prompt = "";
        g_sentence_accumulator = ""; // 清空缓冲区
    }
    napi_value result;
    napi_create_int32(env, 1, &result);
    return result;
}

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"nativeLoad", nullptr, NativeLoad, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"nativeChat", nullptr, NativeChat, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getLlmResult", nullptr, GetLlmResult, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"initSherpa", nullptr, InitSherpa, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"acceptWaveform", nullptr, AcceptWaveform, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"resetSherpa", nullptr, ResetSherpa, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getRecognizedText", nullptr, GetRecognizedText, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getQueueSize", nullptr, GetQueueSize, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"initTts", nullptr, InitTts, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getTtsAudio", nullptr, GetTtsAudio, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"stopTts", nullptr, StopTts, nullptr, nullptr, nullptr, napi_default, nullptr}
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