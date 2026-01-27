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
  @State currentReply: string = "等待输入..."; // 统一显示 ASR 或 LLM 的内容

  scroller: Scroller = new Scroller();
  private tcpServer: socket.TCPSocketServer | null = null;
  private activeClient: socket.TCPSocketConnection | null = null;
  private context = getContext(this) as common.UIAbilityContext;

  // --- 模型路径 ---
  private asrModelPath: string = this.context.filesDir + "/sherpa_model";
  private llmModelPath: string = this.context.filesDir + "/model.gguf";

  // --- 核心控制变量 ---
  private pollTimer: number = -1;
  private isClientDoneSpeaking: boolean = false; // 客户端发送 [VOICE_END] 标记
  private sessionTranscript: string = "";        // 当前识别到的完整句子
  private isThinking: boolean = false;           // 标记 LLM 是否正在生成

  async aboutToAppear(): Promise<void> {
    this.getIpAddress();

    // 1. 初始化两个引擎
    this.initASR();
    this.initLLM();

    // 2. 启动 TCP 服务
    this.startTCPServer();
    this.addLog("✅ 双模服务就绪 (语音+文字)");

    // 3. 启动统一轮询心跳 (50ms)
    // 替代了 Taskpool，通过查询 C++ 状态来更新 UI
    this.pollTimer = setInterval(() => {
      this.pollSystemStatus();
    }, 50);
  }

  // =============================================================
  // 🔥 核心循环：统一处理 ASR 和 LLM 的状态 🔥
  // =============================================================
  pollSystemStatus() {
    try {
      const lib: ESObject = MNNNamespace;

      // --- [1] ASR 部分：听 ---
      // 只要 C++ 有识别结果，就拿出来显示，但不一定发给客户端
      if (lib.getRecognizedText) {
        let text = lib.getRecognizedText() as string;
        if (text && text.length > 0 && text !== this.sessionTranscript) {
          this.sessionTranscript = text;
          // 如果没在思考，就显示听到的内容
          if (!this.isThinking) {
            this.currentReply = "👂 听取中: " + text;
          }
        }
      }

      // --- [2] LLM 部分：说 ---
      // 如果正在推理，不断轮询生成的 Token
      if (this.isThinking && lib.getLlmResult) {
        let token = lib.getLlmResult() as string;
        if (token && token.length > 0) {
          // 第一次收到 Token 时清除 "思考中..."
          if (this.currentReply === "🤖 思考中...") this.currentReply = "";

          this.currentReply += token; // 更新本地 UI

          // 实时流式回传给客户端
          if (this.activeClient) {
            this.sendPacket(this.activeClient, 0x01, token);
          }
        }
      }

      // --- [3] 逻辑判定：ASR 是否真正结束 ---
      // 条件：客户端发了结束符 AND C++后台队列处理完毕
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
    this.isClientDoneSpeaking = false; // 重置标记

    // 如果是空语音，直接重置
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
    if (this.llmStatus.includes("✅") && lib.nativeChat) {
      this.isThinking = true;
      this.currentReply = "🤖 思考中...";
      // 非阻塞调用，C++ 后台线程开始跑，我们通过 pollSystemStatus 取结果
      lib.nativeChat(query);
    } else {
      this.addLog("⚠️ LLM 未就绪，忽略提问");
      if(this.activeClient) this.sendPacket(this.activeClient, 0x01, "[ERROR] LLM Not Ready");
    }
  }

  // =============================================================
  // 📡 TCP Server (包含旧版的粘包处理 + 新版的接口调用)
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

          // 接收缓冲区
          let buffer = new Uint8Array(0);

          clientSock.on('message', async (val) => {
            if (val.message instanceof ArrayBuffer) {
              // 1. 拼接新数据到缓冲区
              let newPart = new Uint8Array(val.message);
              let temp = new Uint8Array(buffer.length + newPart.length);
              temp.set(buffer, 0);
              temp.set(newPart, buffer.length);
              buffer = temp;

              // 2. 循环处理缓冲区中的完整包
              while (buffer.length >= 5) { // 至少要有头部 (1字节类型 + 4字节长度)
                let view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                let type = view.getUint8(0);
                let len = view.getUint32(1);
                let packetLen = 5 + len;

                // 数据不够一个完整包，等待下一次
                if (buffer.length < packetLen) break;

                // 提取包体
                let body = buffer.subarray(5, packetLen);

                // --- 业务逻辑处理 ---
                if (type === 0x01) { // [纯文本]
                  let text = new util.TextDecoder().decode(body);
                  if (text.trim() === 'ping') {
                    // 可以选择不打印日志，以免刷屏
                    // this.addLog("❤️ 心跳");
                  }else if (text.includes("[VOICE_END]")) {
                    this.addLog("🎤 收到语音结束符");
                    // 标记结束，剩下的交给轮询器去查 C++ 队列
                    this.isClientDoneSpeaking = true;
                  } else {
                    // 普通文字聊天：直接触发 LLM
                    this.addLog(`📨 收到文字: ${text}`);
                    // 打断当前的语音识别（如果有）
                    const lib: ESObject = MNNNamespace;
                    if (lib.resetSherpa) lib.resetSherpa();
                    this.sessionTranscript = "";

                    this.triggerLLM(text);
                  }

                } else if (type === 0x02) { // [语音流]
                  const lib: ESObject = MNNNamespace;
                  if (lib.acceptWaveform) {
                    // 直接塞入 C++ 队列，不做任何 JS 层缓冲，速度最快
                    lib.acceptWaveform(body.slice(0).buffer);
                  }
                }

                // 移除已处理的包
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
      this.addLog("⏳ 初始化 LLM (后台线程)...");
      if (lib.nativeLoad) {
        // 显式类型断言，防止 ArkTS 报错
        let ret = lib.nativeLoad(this.llmModelPath) as boolean;
        if (ret) {
          this.llmStatus = "✅ LLM 就绪";
          this.addLog("🧠 大模型加载完成");
        } else {
          this.llmStatus = "❌ LLM 加载失败";
        }
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
        // 显式类型断言
        let ret = lib.initSherpa(this.asrModelPath) as boolean;
        if (ret) this.asrStatus = "✅ Sherpa 就绪";
      }
    } catch (e) { this.asrStatus = "❌ Init Error"; }
  }

  sendPacket(client: socket.TCPSocketConnection, type: number, data: string) {
    if (!client) return;
    let contentBytes = new util.TextEncoder().encodeInto(data);
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

  // =============================================================
  // 🎨 UI 布局
  // =============================================================
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
      }.margin({bottom: 10})

      // 日志区
      List({ scroller: this.scroller }) {
        ForEach(this.logs, (log: string) => {
          ListItem() { Text(log).fontSize(11).fontColor('#666').margin({bottom:2}) }
        })
      }.height('25%').width('95%').backgroundColor('#F2F2F2').borderRadius(5)

      // 交互显示区 (ASR实时结果 或 LLM回答)
      Text(this.currentReply)
        .fontSize(22).fontColor('#333').fontWeight(FontWeight.Bold)
        .padding(15).width('95%').layoutWeight(1)
        .backgroundColor('#E6F7FF').margin({top:10}).borderRadius(10)
        .align(Alignment.TopStart)

      // 手动测试按钮
      Button("手动测试: 你好")
        .onClick(() => {
          this.addLog("🖱️ 手动触发");
          this.triggerLLM("你好");
        })
        .margin(10)
    }
    .width('100%').height('100%')
  }
}