import { contextBridge, ipcRenderer } from 'electron';

import type { AiInstructionSourceSnapshot } from '../common/aiInstructionTypes';
import type { ArcProjectBrowserSnapshot } from '../common/projectTypes';
import type { AiRuntimeStreamEnvelope, AiRuntimeStreamStartRequest } from '../common/aiRuntimeIpcTypes';

export type ArcAiRuntimeApi = {
  start(request: AiRuntimeStreamStartRequest): Promise<void>;
  cancel(requestId: string): Promise<boolean>;
  instructionSources(): Promise<AiInstructionSourceSnapshot>;
  onEvent(callback: (event: AiRuntimeStreamEnvelope) => void): () => void;
};

const api: ArcAiRuntimeApi = {
  start: (request) => ipcRenderer.invoke('ai-runtime:start', request),
  cancel: (requestId) => ipcRenderer.invoke('ai-runtime:cancel', requestId),
  instructionSources: async () => {
    const snapshot = (await ipcRenderer.invoke('project:snapshot')) as ArcProjectBrowserSnapshot | null;
    const project = snapshot?.activeProject;
    return ipcRenderer.invoke(
      'ai-runtime:instruction-sources',
      project
        ? {
            projectRoot: project.projectRoot,
            descriptorPath: project.descriptorPath,
            projectGuid: project.descriptor.guid,
          }
        : null,
    );
  },
  onEvent: (callback) => {
    const listener = (_event: Electron.IpcRendererEvent, envelope: AiRuntimeStreamEnvelope) => callback(envelope);
    ipcRenderer.on('ai-runtime:event', listener);
    return () => ipcRenderer.removeListener('ai-runtime:event', listener);
  },
};

contextBridge.exposeInMainWorld('arcAiRuntime', api);
