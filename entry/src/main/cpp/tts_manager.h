#pragma once
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "sherpa-ncnn/csrc/offline-tts.h"

class TtsManager {
public:
    static TtsManager& Instance() {
        static TtsManager instance;
        return instance;
    }

    // 初始化模型
    bool Init(const std::string& modelPath);
    
    // 输入待合成文本（由 LLM 线程调用）
    void PushText(const std::string& text);
    
    // 获取合成好的音频数据（由 JS 轮询调用）
    std::vector<int16_t> PopAudio();

    // 停止并清理（打断机制）
    void Stop();

private:
    TtsManager() : g_running(false), tts_thread(nullptr) {}
    void WorkingThread();

    sherpa_ncnn::OfflineTts* tts = nullptr;
    std::queue<std::string> text_queue;
    std::queue<std::vector<int16_t>> audio_queue;
    
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> g_running;
    std::thread* tts_thread;
};