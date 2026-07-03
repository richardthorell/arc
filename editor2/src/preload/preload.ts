import { contextBridge, ipcRenderer } from 'electron';

export type ArcStartupState = {
  appVersion: string;
  engineHostConnected: boolean;
  viewportMode: 'placeholder' | 'native' | 'streamed';
};

const arcApi = {
  getVersion: (): Promise<string> => ipcRenderer.invoke('app:getVersion'),
  getStartupState: (): Promise<ArcStartupState> => ipcRenderer.invoke('editor:getStartupState'),
};

contextBridge.exposeInMainWorld('arc', arcApi);

export type ArcApi = typeof arcApi;
