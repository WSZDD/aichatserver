#include "sherpa_napi.h"
#include "sherpa-ncnn/sherpa-ncnn/c-api/c-api.h"
#include <hilog/log.h>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <stdlib.h> // for setenv

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x0000
#define LOG_TAG "SHERPA_TURBO" // 改个名字代表极速版
#define LOGI(...) OH_LOG_Print(LOG_APP, LOG_INFO, LOG_DOMAIN, LOG_TAG, __VA_ARGS__)

static SherpaNcnnRecognizer *g_recognizer = nullptr;
static SherpaNcnnStream *g_stream = nullptr;
static std::mutex g_data_mutex;
static std::deque<float> g_audio_buffer;
static std::string g_result_buffer = "";
static std::atomic<bool> g_running = false;
static std::thread* g_worker_thread = nullptr;

// 🔥 后台线程：全速计算 🔥
void BackgroundWorker() {
    LOGI("🧵 后台线程启动 (Turbo Mode)");
    
    while (g_running) {
        std::vector<float> samples;
        int queue_size = 0;

        {
            std::lock_guard<std::mutex> lock(g_data_mutex);
            queue_size = g_audio_buffer.size();
            
            // 每次取 0.4s (6400点)
            // 如果积压严重 (>1秒)，就多取一点(0.8s)来追赶进度
            int target_fetch = (queue_size > 16000) ? 12800 : 6400;
            int fetch_size = std::min(queue_size, target_fetch); 
            
            if (fetch_size > 0) {
                for(int i=0; i<fetch_size; i++) {
                    samples.push_back(g_audio_buffer.front());
                    g_audio_buffer.pop_front();
                }
            }
        }

        if (samples.empty()) {
            usleep(5000); // 没数据睡 5ms
            continue;
        }

        // --- 性能计时 ---
        auto start = std::chrono::high_resolution_clock::now();

        if (g_recognizer && g_stream) {
            AcceptWaveform(g_stream, 16000, samples.data(), samples.size());
            
            while (IsReady(g_recognizer, g_stream)) {
                Decode(g_recognizer, g_stream);
            }
            
            SherpaNcnnResult* result = GetResult(g_recognizer, g_stream);
            std::string text = result->text;
            DestroyResult(result);

            if (!text.empty()) {
                std::lock_guard<std::mutex> lock(g_data_mutex);
                g_result_buffer = text;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        // 只打印耗时较长的日志，避免刷屏
        if (duration > 200) {
             LOGI("⚡ 耗时: %{public}lldms | 积压: %{public}d", duration, queue_size);
        }
    }
}

napi_value InitSherpa(napi_env env, napi_callback_info info) {
    std::lock_guard<std::mutex> lock(g_data_mutex);
    if (g_recognizer) {
        napi_value res; napi_get_boolean(env, true, &res); return res;
    }

    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    char pathBuf[512];
    size_t strSize;
    napi_get_value_string_utf8(env, args[0], pathBuf, 512, &strSize);
    std::string modelDir = pathBuf;

    // 🔥 1. 环境变量优化 (32位系统专用) 🔥
    setenv("NCNN_USE_FP16_PACKED", "0", 1);
    setenv("NCNN_USE_FP16_STORAGE", "0", 1);
    setenv("NCNN_USE_FP16_ARITHMETIC", "0", 1); // 禁用FP16，防止软解卡死
    setenv("NCNN_CPU_POWERSAVE", "0", 1);       // 绑定大核
    setenv("OMP_NUM_THREADS", "2", 1);          // 配合下面的 num_threads

    SherpaNcnnRecognizerConfig config;
    memset(&config, 0, sizeof(config)); 
    
    // 🔥 2. 开启双线程 (Int8模型在2线程下更快) 🔥
    config.model_config.num_threads = 2; 

    std::string tokens = modelDir + "/tokens.txt";
    std::string encoder_bin = modelDir + "/encoder_jit_trace-pnnx.ncnn.bin";
    std::string encoder_param = modelDir + "/encoder_jit_trace-pnnx.ncnn.param";
    std::string decoder_bin = modelDir + "/decoder_jit_trace-pnnx.ncnn.bin";
    std::string decoder_param = modelDir + "/decoder_jit_trace-pnnx.ncnn.param";
    std::string joiner_bin = modelDir + "/joiner_jit_trace-pnnx.ncnn.bin";
    std::string joiner_param = modelDir + "/joiner_jit_trace-pnnx.ncnn.param";

    config.model_config.tokens = tokens.c_str();
    config.model_config.encoder_bin = encoder_bin.c_str();
    config.model_config.encoder_param = encoder_param.c_str();
    config.model_config.decoder_bin = decoder_bin.c_str();
    config.model_config.decoder_param = decoder_param.c_str();
    config.model_config.joiner_bin = joiner_bin.c_str();
    config.model_config.joiner_param = joiner_param.c_str();
    
    config.decoder_config.decoding_method = "greedy_search";
    
    // 🔥 3. 限制解码搜索路径 (性能提升关键) 🔥
    // 默认值很大，改为 4 可以显著减少 CPU 负担，对精度影响微乎其微
    config.decoder_config.num_active_paths = 4; 

    // 🔥 4. 调整 VAD 灵敏度 (跳过静音) 🔥
    config.enable_endpoint = 1; // 开启端点检测算法
    config.rule1_min_trailing_silence = 1.2f; 
    config.rule2_min_trailing_silence = 0.8f; 

    config.feat_config.sampling_rate = 16000;
    config.feat_config.feature_dim = 80;

    g_recognizer = CreateRecognizer(&config);
    if (g_recognizer) {
        g_stream = CreateStream(g_recognizer);
        LOGI("✅ Sherpa Init OK (Threads=2, Paths=4)");
        if (!g_running) {
            g_running = true;
            g_worker_thread = new std::thread(BackgroundWorker);
            g_worker_thread->detach();
        }
    }
    napi_value res;
    napi_get_boolean(env, true, &res);
    return res;
}

// 生产者：只负责入队
napi_value AcceptWaveform(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    void* data = nullptr;
    size_t len = 0;
    napi_get_arraybuffer_info(env, args[0], &data, &len);

    if (len > 0) {
        int16_t* pcm16 = (int16_t*)data;
        int count = len / 2;
        std::lock_guard<std::mutex> lock(g_data_mutex);
        for(int i=0; i<count; ++i) {
            g_audio_buffer.push_back(pcm16[i] / 32768.0f);
        }
    }
    napi_value res;
    napi_create_string_utf8(env, "", 0, &res);
    return res;
}

// 消费者：JS 轮询
napi_value GetRecognizedText(napi_env env, napi_callback_info info) {
    std::string res = "";
    {
        std::lock_guard<std::mutex> lock(g_data_mutex);
        res = g_result_buffer;
    }
    napi_value output;
    napi_create_string_utf8(env, res.c_str(), NAPI_AUTO_LENGTH, &output);
    return output;
}

// 手动重置
napi_value ResetSherpa(napi_env env, napi_callback_info info) {
    std::lock_guard<std::mutex> lock(g_data_mutex);
    if (g_recognizer && g_stream) {
        Reset(g_recognizer, g_stream);
    }
    g_result_buffer = "";
    g_audio_buffer.clear();
    LOGI("🔄 Manual Reset Done");
    return nullptr;
}

// 查岗接口
napi_value GetQueueSize(napi_env env, napi_callback_info info) {
    int size = 0;
    {
        std::lock_guard<std::mutex> lock(g_data_mutex);
        size = (int)g_audio_buffer.size();
    }
    napi_value result;
    napi_create_int32(env, size, &result);
    return result;
}