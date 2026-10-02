import { contextBridge, ipcRenderer } from 'electron';

import type { AiRuntimeStreamEnvelope, AiRuntimeStreamStartRequest } from '../common/aiRuntimeIpcTypes';

export type ArcAiRuntimeApi = {
  start(request: AiRuntimeStreamStartRequest): Promise<void>;
  cancel(requestId: string): Promise<boolean>;
  onEvent(callback: (event: AiRuntimeStreamEnvelope) => void): () => void;
};

const api: ArcAiRuntimeApi = {
  start: (request) => ipcRenderer.invoke('ai-runtime:start', request),
  cancel: (requestId) => ipcRenderer.invoke('ai-runtime:cancel', requestId),
  onEvent: (callback) => {
    const listener = (_event: Electron.IpcRendererEvent, envelope: AiRuntimeStreamEnvelope) => callback(envelope);
    ipcRenderer.on('ai-runtime:event', listener);
    return () => ipcRenderer.removeListener('ai-runtime:event', listener);
  },
};

contextBridge.exposeInMainWorld('arcAiRuntime', api);
