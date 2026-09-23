import { contextBridge, ipcRenderer } from 'electron';

import type { AiProviderAccountsSnapshot, AiProviderId } from '../common/aiProviderTypes';
import './preload';

const arcAiProvidersApi = {
  snapshot: (): Promise<AiProviderAccountsSnapshot> => ipcRenderer.invoke('ai-providers:snapshot'),
  connect: (providerId: AiProviderId, credential: string): Promise<AiProviderAccountsSnapshot> =>
    ipcRenderer.invoke('ai-providers:connect', providerId, credential),
  disconnect: (providerId: AiProviderId): Promise<AiProviderAccountsSnapshot> =>
    ipcRenderer.invoke('ai-providers:disconnect', providerId),
};

contextBridge.exposeInMainWorld('arcAiProviders', arcAiProvidersApi);

export type ArcAiProvidersApi = typeof arcAiProvidersApi;
