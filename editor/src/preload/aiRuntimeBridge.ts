import { contextBridge, ipcRenderer } from 'electron';

import type {
  BuiltInAgentEvent,
  BuiltInAgentInvokeRequest,
  BuiltInAgentRuntimeBridge,
  BuiltInAgentToolInvokeRequest,
} from '../common/builtInAgentTypes';
import type { AiJsonObject } from '../common/aiRuntimeTypes';
import type { AiInstructionSourceSnapshot } from '../common/aiInstructionTypes';
import type { ArcProjectBrowserSnapshot } from '../common/projectTypes';
import type { AiRuntimeStreamEnvelope, AiRuntimeStreamStartRequest } from '../common/aiRuntimeIpcTypes';

export type ArcAiRuntimeApi = {
  start(request: AiRuntimeStreamStartRequest): Promise<void>;
  cancel(requestId: string): Promise<boolean>;
  instructionSources(): Promise<AiInstructionSourceSnapshot>;
  onEvent(callback: (event: AiRuntimeStreamEnvelope) => void): () => void;
  agent: BuiltInAgentRuntimeBridge;
};

let agentEventListenerCount = 0;

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
  agent: {
    capabilities: () => ipcRenderer.invoke('ai-runtime:agent-capabilities'),
    invoke: (method, params) => {
      const request: BuiltInAgentInvokeRequest = {
        method,
        ...(params !== undefined ? { params } : {}),
      };
      return ipcRenderer.invoke('ai-runtime:agent-invoke', request);
    },
    tools: () => ipcRenderer.invoke('ai-runtime:agent-tools'),
    invokeTool: (name, arguments_?: AiJsonObject) => {
      const request: BuiltInAgentToolInvokeRequest = {
        name,
        ...(arguments_ !== undefined ? { arguments: arguments_ } : {}),
      };
      return ipcRenderer.invoke('ai-runtime:agent-invoke-tool', request);
    },
    onEvent: (callback) => {
      const listener = (_event: Electron.IpcRendererEvent, agentEvent: BuiltInAgentEvent) => callback(agentEvent);
      ipcRenderer.on('ai-runtime:agent-event', listener);
      agentEventListenerCount += 1;
      if (agentEventListenerCount === 1) void ipcRenderer.invoke('ai-runtime:agent-subscribe').catch(() => undefined);
      return () => {
        ipcRenderer.removeListener('ai-runtime:agent-event', listener);
        agentEventListenerCount = Math.max(0, agentEventListenerCount - 1);
        if (agentEventListenerCount === 0)
          void ipcRenderer.invoke('ai-runtime:agent-unsubscribe').catch(() => undefined);
      };
    },
  },
};

contextBridge.exposeInMainWorld('arcAiRuntime', api);
