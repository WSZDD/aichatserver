export interface MNNNativeAPI {
  nativeLoad(modelPath: string): boolean;
  nativeChat(question: string, onToken: (token: string) => void): string;
}

declare module 'libmnnllm.so' {
  const nativeExport: MNNNativeAPI & { default: MNNNativeAPI };
  export default nativeExport;
}