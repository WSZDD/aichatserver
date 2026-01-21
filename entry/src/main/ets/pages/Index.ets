import socket from '@ohos.net.socket';
import wifiManager from '@ohos.wifiManager';
import util from '@ohos.util';
import fs from '@ohos.file.fs';
import common from '@ohos.app.ability.common';
import { BusinessError } from '@ohos.base';
import promptAction from '@ohos.promptAction';
import taskpool from '@ohos.taskpool';
import emitter from '@ohos.events.emitter';

// 引入库
import MNNNamespace from 'libmnnllm.so';

// ====================================================================
// 🧵 后台任务定义 (保持不变)
// ====================================================================

// 1. 加载任务
@Concurrent
function nativeLoadTask(modelPath: string): boolean {
  const lib: ESObject = MNNNamespace;
  // 安全调用
  if (lib.nativeLoad) {
    return lib.nativeLoad(modelPath);
  }
  if (lib.default && lib.default.nativeLoad) {
    return lib.default.nativeLoad(modelPath);
  }
  return false;
}

// 2. 推理任务
@Concurrent
function nativeChatTask(question: string): string {
  const lib: ESObject = MNNNamespace;

  // 回调：C++ 每生成一个字，就通过 Emitter 发出来
  const onTokenCallback = (token: string): void => {
    emitter.emit({ eventId: 100 }, {
      data: { token: token }
    });
  };

  if (lib.nativeChat) {
    return lib.nativeChat(question, onTokenCallback);
  }
  if (lib.default && lib.default.nativeChat) {
    return lib.default.nativeChat(question, onTokenCallback);
  }
  return "Error";
}

// ====================================================================
// 📱 主界面 (服务端)
// ====================================================================

@Entry
@Component
struct Index {
  @State logs: string[] = ["⏳ 系统初始化..."];
  @State ipAddress: string = "获取中...";
  @State modelStatus: string = "未加载";
  @State clientStatus: string = "等待连接...";
  @State isModelLoading: boolean = false;
  @State currentReply: string = "";

  scroller: Scroller = new Scroller();
  private tcpServer: socket.TCPSocketServer | null = null;
  private activeClient: socket.TCPSocketConnection | null = null;

  // 显式类型转换
  private context = getContext(this) as common.UIAbilityContext;
  private modelPath: string = this.context.filesDir + "/model.gguf";

  async aboutToAppear(): Promise<void> {
    this.getIpAddress();
    this.checkModelFile();

    // ✅ 核心修复：给 eventData 添加显式类型 emitter.EventData
    emitter.on({ eventId: 100 }, (eventData: emitter.EventData) => {
      // 1. 使用 ESObject 接收 data，避开类型检查
      const rawData: ESObject = eventData.data;

      if (rawData && rawData.token) {
        const token = rawData.token as string;

        // 2. 更新本地屏幕
        this.currentReply += token;

        // 3. 实时转发给客户端
        if (this.activeClient) {
          this.activeClient.send({ data: token }).catch((err: BusinessError) => {
            // 忽略发送错误
          });
        }
      }
    });

    setTimeout(() => { this.initModel(); }, 1000);
  }

  checkModelFile(): void {
    try {
      if (fs.accessSync(this.modelPath)) {
        let stat = fs.statSync(this.modelPath);
        this.addLog(`✅ 模型就绪: ${(stat.size/1024/1024).toFixed(2)} MB`);
      } else {
        this.addLog(`❌ 模型丢失: ${this.modelPath}`);
      }
    } catch (e) { this.addLog(`Check Error: ${e}`); }
  }

  async initModel(): Promise<void> {
    if (this.isModelLoading) return;
    this.isModelLoading = true;
    this.addLog("📥 后台加载模型中...");

    try {
      if (!fs.accessSync(this.modelPath)) {
        this.modelStatus = "❌ 文件丢失";
        this.isModelLoading = false;
        return;
      }
      let sT = new Date().getTime();
      let task = new taskpool.Task(nativeLoadTask, this.modelPath);
      let success = await taskpool.execute(task) as boolean;

      if (success) {
        this.modelStatus = "✅ 模型已加载";
        this.addLog(`🎉 加载耗时: ${new Date().getTime() - sT}ms`);
        this.startTCPServer();
      } else {
        this.modelStatus = "❌ 加载失败";
      }
    } catch (e) {
      this.modelStatus = "❌ 崩溃";
      this.addLog("Init Error: " + JSON.stringify(e));
    } finally {
      this.isModelLoading = false;
    }
  }

  async runChat(question: string): Promise<void> {
    this.currentReply = "";
    this.addLog("🤖 AI 推理中...");

    try {
      let task = new taskpool.Task(nativeChatTask, question);
      // 等待 C++ 推理任务彻底结束
      await taskpool.execute(task);
      this.addLog("✅ 回复完成");

      // ✨【新增】推理结束后，发送结束标记给客户端
      if (this.activeClient) {
        // 发送一个特殊的结束符，客户端收到后会隐藏光标
        this.activeClient.send({ data: "[END]" }).catch(() => {});
      }

    } catch (e) {
      this.addLog("❌ 推理错误: " + JSON.stringify(e));
    }
  }

  startTCPServer(): void {
    if (this.tcpServer) return;
    this.tcpServer = socket.constructTCPSocketServerInstance();

    class MyNetAddress implements socket.NetAddress {
      address: string = '0.0.0.0';
      port: number = 8765;
      family: number = 1;
    }
    let addr: socket.NetAddress = new MyNetAddress();

    this.tcpServer.listen(addr, (err: BusinessError) => {
      if (err) { this.addLog("端口 8765 被占用"); return; }
      this.addLog("🚀 TCP 服务启动");

      if(this.tcpServer) {
        this.tcpServer.on('connect', (clientSock) => {
          this.clientStatus = "📱 客户端已连接";
          this.addLog("🔔 客户端接入");

          this.activeClient = clientSock;

          clientSock.on('message', async (val) => {
            if (val.message instanceof ArrayBuffer) {
              let textDecoder = util.TextDecoder.create('utf-8');
              // 解码文本
              let text = textDecoder.decode(new Uint8Array(val.message));

              // 🛑 【核心修复】在此处拦截心跳包
              // 如果收到的是 "ping"，直接忽略或仅打印日志，绝对不要调用 runChat
              if (text.trim() === 'ping') {
                // this.addLog("💓 收到心跳"); // 如果觉得日志太烦，这行也可以注释掉
                return; // 👈 关键：直接结束函数，不往下执行
              }

              this.addLog("👤 收到指令: " + text);

              if (this.modelStatus.includes("✅")) {
                // 只有不是心跳包，才让 AI 回复
                await this.runChat(text);
              } else {
                clientSock.send({ data: "Server Model Not Ready" });
              }
            }
          });

          clientSock.on('close', () => {
            this.clientStatus = "等待连接...";
            this.addLog("📴 客户端断开");
            this.activeClient = null;
          });

          clientSock.on('error', () => {
            this.activeClient = null;
          });
        });
      }
    });
  }

  getIpAddress(): void {
    try {
      let ipInfo = wifiManager.getIpInfo();
      let ip = ipInfo.ipAddress;
      this.ipAddress = ((ip>>>24)&0xFF)+"."+((ip>>16)&0xFF)+"."+((ip>>8)&0xFF)+"."+(ip&0xFF);
    } catch (e) {}
  }

  addLog(msg: string): void {
    console.info("[LLM_SERVER] " + msg);
    let time = new Date().toLocaleTimeString();
    this.logs.push(`[${time}] ${msg}`);
    setTimeout(() => { this.scroller.scrollEdge(Edge.Bottom); }, 100);
  }

  build() {
    Column() {
      Text("RK3588 端侧大模型 (服务端)").fontSize(20).fontWeight(FontWeight.Bold).margin({ top: 20 })

      Row() {
        Text(this.modelStatus).fontColor(this.modelStatus.includes("✅") ? '#00AA00' : 'red').margin({right:10})
        Text(this.ipAddress + ":8765").fontColor('blue')
      }.margin(10)

      List({ scroller: this.scroller }) {
        ForEach(this.logs, (log: string) => {
          ListItem() { Text(log).fontSize(12).width('100%').fontColor('#666') }
        })
      }.height('25%').width('95%').backgroundColor('#F0F0F0').borderRadius(5)

      Divider().margin(10)

      // 显示当前的实时回复
      Scroll() {
        Text(this.currentReply)
          .fontSize(18)
          .fontColor('#333')
          .padding(10)
      }
      .layoutWeight(1)
      .width('95%')
      .backgroundColor('#E6F7FF')
      .borderRadius(10)
      .align(Alignment.TopStart)

      Button(this.isModelLoading ? "加载中..." : "手动测试: 你好")
        .onClick(async () => {
          if(!this.isModelLoading && this.modelStatus.includes("✅")) {
            await this.runChat("你好");
          } else {
            this.initModel();
          }
        })
        .margin(10)
    }
    .width('100%').height('100%')
  }
}