import { contextBridge, ipcRenderer } from 'electron';

export type ArcStartupState = {
  appVersion: string;
  engineHostConnected: boolean;
  viewportMode: 'placeholder' | 'native' | 'streamed';
};

const arcApi = {
  getVersion: (): Promise<string> => ipcRenderer.invoke('app:getVersion'),
  getStartupState: (): Promise<ArcStartupState> => ipcRenderer.invoke('editor:getStartupState'),
  nativeWindow: {
    minimize: (): Promise<void> => ipcRenderer.invoke('nativeWindow:minimize'),
    toggleMaximize: (): Promise<boolean> => ipcRenderer.invoke('nativeWindow:toggleMaximize'),
    close: (): Promise<void> => ipcRenderer.invoke('nativeWindow:close'),
    isMaximized: (): Promise<boolean> => ipcRenderer.invoke('nativeWindow:isMaximized'),
    onMaximizedChanged: (callback: (maximized: boolean) => void): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, maximized: boolean) => callback(maximized);
      ipcRenderer.on('nativeWindow:maximizedChanged', listener);
      return () => ipcRenderer.removeListener('nativeWindow:maximizedChanged', listener);
    },
  },
};

contextBridge.exposeInMainWorld('arc', arcApi);

export type ArcApi = typeof arcApi;
