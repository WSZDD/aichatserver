#include "tts_manager.h"
// 必须引用的头文件
#include "sherpa-ncnn/csrc/offline-tts.h"
#include "sherpa-ncnn/csrc/offline-tts-model-config.h"
#include "sherpa-ncnn/csrc/offline-tts-vits-model-config.h" 

#include <hilog/log.h>
#include <thread>
#include <mutex>
#include <vector>
#include <deque>
#include <string>
#include <atomic>
#include <unistd.h>
#include <stdlib.h> 

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x0000
#define LOG_TAG "SHERPA_TTS"
#define LOGI(...) OH_LOG_Print(LOG_APP, LOG_INFO, LOG_DOMAIN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_DOMAIN, LOG_TAG, __VA_ARGS__)

// ==========================================
// 全局静态资源
// ==========================================
static sherpa_ncnn::OfflineTts* g_tts = nullptr;
static std::mutex g_tts_mutex;
static std::deque<std::string> g_text_queue;       
static std::deque<int16_t> g_pcm_buffer;           
static std::atomic<bool> g_tts_running = false;
static std::thread* g_tts_thread = nullptr;

// ==========================================
// 后台线程
// ==========================================
void TtsBackgroundWorker() {
    LOGI("🧵 TTS 后台线程启动 (ModelDir Mode)");
    
    // RK3568 32位系统优化环境变量 (必加!)
    setenv("NCNN_USE_FP16_PACKED", "0", 1);
    setenv("NCNN_USE_FP16_STORAGE", "0", 1);
    setenv("NCNN_USE_FP16_ARITHMETIC", "0", 1); 
    setenv("NCNN_CPU_POWERSAVE", "0", 1);
    setenv("OMP_NUM_THREADS", "1", 1); 

    while (g_tts_running) {
        std::string current_text = "";
        
        {
            std::lock_guard<std::mutex> lock(g_tts_mutex);
            if (!g_text_queue.empty()) {
                current_text = g_text_queue.front();
                g_text_queue.pop_front();
            }
        }

        if (current_text.empty()) {
            usleep(20000); 
            continue;
        }

        if (g_tts) {
            // 使用 TtsArgs 传参
            sherpa_ncnn::TtsArgs args;
            args.text = current_text;
            args.sid = 0;      
            args.speed = 1.2f; 
            
            auto audio = g_tts->Generate(args);
            
            if (!audio.samples.empty()) {
                std::lock_guard<std::mutex> lock(g_tts_mutex);
                
                // float -> int16
                for (float s : audio.samples) {
                    if (s > 1.0f) s = 1.0f;
                    if (s < -1.0f) s = -1.0f;
                    int16_t val = static_cast<int16_t>(s * 32767.0f);
                    g_pcm_buffer.push_back(val);
                }
            }
        }
    }
    LOGI("🛑 TTS 线程退出");
}

// ==========================================
// TtsManager 实现
// ==========================================

bool TtsManager::Init(const std::string& modelPath) {
    std::lock_guard<std::mutex> lock(g_tts_mutex);
    if (g_tts) return true;

    sherpa_ncnn::OfflineTtsConfig config;

    // ==========================================
    // 🔥 核心修正：只设置 model_dir 🔥
    // ==========================================
    // 根据你提供的头文件，OfflineTtsVitsModelConfig 只有一个 model_dir 成员
    // 它会自动在目录下查找 config.json, lexicon.txt 等文件
    config.model.vits.model_dir = modelPath; 

    // 性能配置
    config.model.num_threads = 1; 
    config.model.debug = 0;

    // 简单校验目录是否存在
    if (access(modelPath.c_str(), F_OK) != 0) {
        LOGE("❌ 模型目录缺失: %{public}s", modelPath.c_str());
        return false;
    }

    try {
        g_tts = new sherpa_ncnn::OfflineTts(config);
        
        if (!g_tts_running) {
            g_tts_running = true;
            g_tts_thread = new std::thread(TtsBackgroundWorker);
            g_tts_thread->detach();
        }
        
        LOGI("✅ TTS Init OK");
        return true;
    } catch (const std::exception &e) {
        LOGE("❌ TTS Init Exception: %{public}s", e.what());
        return false;
    } catch (...) {
        LOGE("❌ TTS Init Unknown Exception");
        return false;
    }
}

void TtsManager::PushText(const std::string& text) {
    if (text.empty()) return;
    std::lock_guard<std::mutex> lock(g_tts_mutex);
    g_text_queue.push_back(text);
}

std::vector<int16_t> TtsManager::PopAudio() {
    std::vector<int16_t> chunk;
    std::lock_guard<std::mutex> lock(g_tts_mutex);
    if (g_pcm_buffer.empty()) return chunk;

    size_t fetch_size = std::min((size_t)8192, g_pcm_buffer.size());
    chunk.reserve(fetch_size);

    for (size_t i = 0; i < fetch_size; ++i) {
        chunk.push_back(g_pcm_buffer.front());
        g_pcm_buffer.pop_front();
    }
    return chunk;
}

void TtsManager::Stop() {
    std::lock_guard<std::mutex> lock(g_tts_mutex);
    g_text_queue.clear();
    g_pcm_buffer.clear();
    LOGI("🚫 TTS Queue Cleared");
}