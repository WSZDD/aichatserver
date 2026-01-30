import socket from '@ohos.net.socket';
import wifiManager from '@ohos.wifiManager';
import util from '@ohos.util';
import fs from '@ohos.file.fs';
import common from '@ohos.app.ability.common';
import { BusinessError } from '@ohos.base';

// 引入 Native 库
import MNNNamespace from 'libmnnllm.so';

@Entry
@Component
struct Index {
  // --- UI 状态 ---
  @State logs: string[] = ["⏳ 系统初始化..."];
  @State ipAddress: string = "获取中...";
  @State asrStatus: string = "ASR: 未加载";
  @State llmStatus: string = "LLM: 未加载";
  @State ttsStatus: string = "TTS: 未加载";
  @State currentReply: string = "等待输入...";

  scroller: Scroller = new Scroller();
  private tcpServer: socket.TCPSocketServer | null = null;
  private activeClient: socket.TCPSocketConnection | null = null;
  private context = getContext(this) as common.UIAbilityContext;

  // --- 模型路径 ---
  // ASR 模型目录
  private asrModelPath: string = this.context.filesDir + "/sherpa_model";
  // LLM 模型文件
  private llmModelPath: string = this.context.filesDir + "/model.gguf";
  // TTS 模型目录
  private ttsModelPath: string = this.context.filesDir + "/sherpa_tts_model";

  // --- 核心控制变量 ---
  private pollTimer: number = -1;
  private isClientDoneSpeaking: boolean = false;
  private sessionTranscript: string = "";
  private isThinking: boolean = false;

  async aboutToAppear(): Promise<void> {
    this.getIpAddress();

    // 1. 初始化三个引擎 (ASR, LLM, TTS)
    this.initASR();
    this.initLLM();
    this.initTTS();

    // 2. 启动 TCP 服务
    this.startTCPServer();
    this.addLog("✅ 全栈服务就绪 (语音识别+大模型+语音合成)");

    // 3. 启动统一轮询心跳 (50ms)
    this.pollTimer = setInterval(() => {
      this.pollSystemStatus();
    }, 50);
  }

  // =============================================================
  // 🔥 核心循环：统一处理 ASR / LLM / TTS 的状态 🔥
  // =============================================================
  pollSystemStatus() {
    try {
      const lib: ESObject = MNNNamespace;

      // --- [1] ASR 部分：听 ---
      if (lib.getRecognizedText) {
        let text = lib.getRecognizedText() as string;
        if (text && text.length > 0 && text !== this.sessionTranscript) {
          this.sessionTranscript = text;
          if (!this.isThinking) {
            this.currentReply = "👂 听取中: " + text;
          }
        }
      }

      // --- [2] LLM 部分：想 (文本流) ---
      if (this.isThinking && lib.getLlmResult) {
        let token = lib.getLlmResult() as string;
        if (token && token.length > 0) {
          if (this.currentReply === "🤖 思考中...") this.currentReply = "";
          this.currentReply += token;

          // 实时回传文本给客户端 UI 显示
          if (this.activeClient) {
            this.sendPacket(this.activeClient, 0x01, token);
          }
        }
      }

      // --- [3] TTS 部分：说 (音频流) ---
      // 🔥 新增：轮询 C++ 产生的 TTS 音频数据
      if (lib.getTtsAudio) {
        // 🔥🔥 核心修复：显式类型断言 as ArrayBuffer，解决 ArkTS 报错 🔥🔥
        let pcmChunk = lib.getTtsAudio() as ArrayBuffer;

        if (pcmChunk && pcmChunk.byteLength > 0 && this.activeClient) {
          // 发送 0x03 类型包 (Audio)
          this.sendPacket(this.activeClient, 0x03, pcmChunk);
        }
      }

      // --- [4] 逻辑判定：ASR 是否真正结束 ---
      if (this.isClientDoneSpeaking && lib.getQueueSize) {
        let pendingSize = lib.getQueueSize() as number;
        if (pendingSize === 0) {
          this.handleAsrFinalResult(lib);
        }
      }
    } catch (e) {}
  }

  // ASR 最终处理：重置语音 -> 触发 LLM
  handleAsrFinalResult(lib: ESObject) {
    const userQuery = this.sessionTranscript;
    this.isClientDoneSpeaking = false;

    if (!userQuery || userQuery.trim().length === 0) {
      if (lib.resetSherpa) lib.resetSherpa();
      return;
    }

    this.addLog(`🗣️ 语音提问: ${userQuery}`);

    // 回传用户说的话 (确认)
    if(this.activeClient) this.sendPacket(this.activeClient, 0x01, `[USER]: ${userQuery}\n`);

    // 触发 LLM
    this.triggerLLM(userQuery);

    // 重置 ASR 引擎
    if (lib.resetSherpa) lib.resetSherpa();
    this.sessionTranscript = "";
  }

  // 统一触发 LLM 推理
  triggerLLM(query: string) {
    const lib: ESObject = MNNNamespace;

    // 🔥 关键：在开始新一轮回答前，强制打断旧的 TTS 播放
    if (lib.stopTts) {
      lib.stopTts();
    }

    if (this.llmStatus.includes("✅") && lib.nativeChat) {
      this.isThinking = true;
      this.currentReply = "🤖 思考中...";
      // C++ 后台线程开始跑，会自动将生成的文本推送到 TTS 队列
      lib.nativeChat(query);
    } else {
      this.addLog("⚠️ LLM 未就绪，忽略提问");
      if(this.activeClient) this.sendPacket(this.activeClient, 0x01, "[ERROR] LLM Not Ready");
    }
  }

  // =============================================================
  // 📡 TCP Server
  // =============================================================
  startTCPServer(): void {
    if (this.tcpServer) return;
    this.tcpServer = socket.constructTCPSocketServerInstance();
    this.tcpServer.listen({ address: '0.0.0.0', port: 8765, family: 1 }, (err: BusinessError) => {
      if (err) return;
      this.addLog("🚀 Server Listening :8765");

      if(this.tcpServer) {
        this.tcpServer.on('connect', (clientSock) => {
          this.activeClient = clientSock;
          this.addLog("🔗 客户端已连接");

          let buffer = new Uint8Array(0);

          clientSock.on('message', async (val) => {
            if (val.message instanceof ArrayBuffer) {
              let newPart = new Uint8Array(val.message);
              let temp = new Uint8Array(buffer.length + newPart.length);
              temp.set(buffer, 0);
              temp.set(newPart, buffer.length);
              buffer = temp;

              while (buffer.length >= 5) {
                let view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                let type = view.getUint8(0);
                let len = view.getUint32(1);
                let packetLen = 5 + len;

                if (buffer.length < packetLen) break;

                let body = buffer.subarray(5, packetLen);

                // --- [1] 纯文本消息 ---
                if (type === 0x01) {
                  let text = new util.TextDecoder().decode(body);
                  if (text.trim() === 'ping') {
                    // 心跳
                  } else if (text.includes("[VOICE_END]")) {
                    this.addLog("🎤 收到语音结束符");
                    this.isClientDoneSpeaking = true;
                  } else {
                    // 文字聊天
                    this.addLog(`📨 收到文字: ${text}`);

                    // 🔥 打断逻辑：用户发文字，立即停止 ASR 和 TTS
                    const lib: ESObject = MNNNamespace;
                    if (lib.resetSherpa) lib.resetSherpa();
                    if (lib.stopTts) lib.stopTts();
                    this.sessionTranscript = "";

                    this.triggerLLM(text);
                  }

                }
                // --- [2] 语音流输入 ---
                else if (type === 0x02) {
                  const lib: ESObject = MNNNamespace;

                  // 🔥 打断逻辑：用户开始说话，立即停止机器人的 TTS
                  if (lib.stopTts) lib.stopTts();

                  if (lib.acceptWaveform) {
                    lib.acceptWaveform(body.slice(0).buffer);
                  }
                }

                buffer = buffer.slice(packetLen);
              }
            }
          });

          clientSock.on('close', () => { this.activeClient = null; this.addLog("客户端断开"); });
          clientSock.on('error', () => { this.activeClient = null; });
        });
      }
    });
  }

  // =============================================================
  // 🛠️ 初始化与工具函数
  // =============================================================

  initLLM() {
    try {
      const lib: ESObject = MNNNamespace;
      if (!fs.accessSync(this.llmModelPath)) {
        this.llmStatus = "❌ model.gguf缺失";
        return;
      }
      this.addLog("⏳ 初始化 LLM...");
      if (lib.nativeLoad) {
        let ret = lib.nativeLoad(this.llmModelPath) as boolean;
        this.llmStatus = ret ? "✅ LLM 就绪" : "❌ LLM 失败";
        if(ret) this.addLog("🧠 大模型加载完成");
      }
    } catch (e) { this.llmStatus = "❌ LLM Error"; }
  }

  initASR() {
    try {
      const lib: ESObject = MNNNamespace;
      if (!fs.accessSync(this.asrModelPath + "/tokens.txt")) {
        this.asrStatus = "❌ ASR模型缺失";
        return;
      }
      if (lib.initSherpa) {
        let ret = lib.initSherpa(this.asrModelPath) as boolean;
        if (ret) this.asrStatus = "✅ Sherpa 就绪";
      }
    } catch (e) { this.asrStatus = "❌ ASR Init Error"; }
  }

  initTTS() {
    try {
      const lib: ESObject = MNNNamespace;

      // 1. 打印当前实际路径，方便调试
      this.addLog("📂 检查路径: " + this.ttsModelPath);

      // 2. 检查目录是否存在
      if (!fs.accessSync(this.ttsModelPath)) {
        this.ttsStatus = "❌ 目录不存在";
        this.addLog("⚠️ 请检查 HDC 推送路径！");
        return;
      }

      // 3. 检查核心文件 config.json
      if (!fs.accessSync(this.ttsModelPath + "/config.json")) {
        this.ttsStatus = "❌ 缺少 config.json";
        return;
      }

      // 4. 🔥 核心修复：自动清洗 tokens.txt 的换行符 🔥
      // Windows 记事本保存的 \r\n 会导致 Sherpa 直接崩溃 (13900012)
      let tokensPath = this.ttsModelPath + "/tokens.txt";
      if (fs.accessSync(tokensPath)) {
        let content = fs.readTextSync(tokensPath);
        if (content.includes('\r')) {
          this.addLog("🔧 修复 tokens.txt 格式...");
          // 替换掉所有 \r，只保留 \n
          let fixedContent = content.replace(/\r/g, '');
          let file = fs.openSync(tokensPath, fs.OpenMode.READ_WRITE | fs.OpenMode.TRUNC);
          fs.writeSync(file.fd, fixedContent);
          fs.closeSync(file);
        }
      } else {
        this.ttsStatus = "❌ 缺少 tokens.txt";
        return;
      }

      // 5. 检查 espeak 数据
      let espeakPath = this.ttsModelPath + "/espeak-ng-data";
      if (!fs.accessSync(espeakPath) || !fs.listFileSync(espeakPath).length) {
        this.ttsStatus = "❌ espeak 缺失/为空";
        return;
      }

      // 6. 一切正常，才敢调用 C++
      this.addLog("🚀 开始加载 C++ TTS...");
      if (lib.initTts) {
        let ret = lib.initTts(this.ttsModelPath) as boolean;
        if (ret) {
          this.ttsStatus = "✅ TTS 就绪";
        } else {
          this.ttsStatus = "❌ C++ 初始化失败";
        }
      }
    } catch (e) {
      this.ttsStatus = "❌ TS层异常";
      console.error("TTS Init Error: " + JSON.stringify(e));
      this.addLog("Error: " + JSON.stringify(e));
    }
  }

  sendPacket(client: socket.TCPSocketConnection, type: number, data: string | ArrayBuffer) {
    if (!client) return;

    let contentBytes: Uint8Array;
    if (typeof data === 'string') {
      contentBytes = new util.TextEncoder().encodeInto(data);
    } else {
      contentBytes = new Uint8Array(data);
    }

    let packet = new Uint8Array(1 + 4 + contentBytes.length);
    let view = new DataView(packet.buffer);
    view.setUint8(0, type);
    view.setUint32(1, contentBytes.length);
    packet.set(contentBytes, 5);

    client.send({ data: packet.buffer as ArrayBuffer }).catch(() => {});
  }

  getIpAddress(): void {
    try {
      let ipInfo = wifiManager.getIpInfo();
      let ip = ipInfo.ipAddress;
      this.ipAddress = ((ip>>>24)&0xFF)+"."+((ip>>16)&0xFF)+"."+((ip>>8)&0xFF)+"."+(ip&0xFF);
    } catch(e) { this.ipAddress = "127.0.0.1"; }
  }

  addLog(msg: string): void {
    if (this.logs.length > 20) this.logs.shift();
    this.logs.push(`[${new Date().toLocaleTimeString()}] ${msg}`);
  }

  build() {
    Column() {
      Text("RK3568 AI 融合服务端").fontSize(18).fontWeight(FontWeight.Bold).margin(15)

      // 状态栏
      Row() {
        Text(this.ipAddress + ":8765").fontColor('blue').fontSize(12)
        Divider().vertical(true).height(15).margin(5)
        Text(this.asrStatus).fontColor(this.asrStatus.includes("✅") ? 'green' : 'red').fontSize(12)
        Divider().vertical(true).height(15).margin(5)
        Text(this.llmStatus).fontColor(this.llmStatus.includes("✅") ? 'green' : 'red').fontSize(12)
        Divider().vertical(true).height(15).margin(5)
        Text(this.ttsStatus).fontColor(this.ttsStatus.includes("✅") ? 'green' : 'red').fontSize(12)
      }.margin({bottom: 10})

      // 日志区
      List({ scroller: this.scroller }) {
        ForEach(this.logs, (log: string) => {
          ListItem() { Text(log).fontSize(11).fontColor('#666').margin({bottom:2}) }
        })
      }.height('25%').width('95%').backgroundColor('#F2F2F2').borderRadius(5)

      // 交互显示区
      Text(this.currentReply)
        .fontSize(22).fontColor('#333').fontWeight(FontWeight.Bold)
        .padding(15).width('95%').layoutWeight(1)
        .backgroundColor('#E6F7FF').margin({top:10}).borderRadius(10)
        .align(Alignment.TopStart)

      // 手动测试按钮
      Row() {
        Button("测试: 你好")
          .onClick(() => {
            this.addLog("🖱️ 手动触发文字");
            this.triggerLLM("你好");
          })
          .margin(5)

        Button("打断说话")
          .backgroundColor('#FF5252')
          .onClick(() => {
            const lib: ESObject = MNNNamespace;
            if (lib.stopTts) {
              lib.stopTts();
              this.addLog("🛑 强制停止 TTS");
            }
          })
          .margin(5)
      }.margin(10)
    }
    .width('100%').height('100%')
  }
}