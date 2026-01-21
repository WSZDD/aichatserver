// 定义一个通用的接口来描述你的 Native 模块
// 这样编译器就知道它有哪些方法，不再瞎猜
export interface MNNModule {
  nativeLoad(path: string): number | boolean;
  nativeLoadAsync(path: string, callback: (res: number) => void): void;
  nativeChat(question: string): string;
  nativeChatAsync(question: string, callback: (res: string) => void): void;
}

// 告诉编译器：import from 'libmnnllm.so' 得到的是 MNNModule
declare module 'libmnnllm.so' {
  const content: MNNModule;
  export default content;
}