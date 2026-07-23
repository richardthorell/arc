import { contextBridge, ipcRenderer } from 'electron';

export type ArcStartupState = {
  appVersion: string;
  engineHostConnected: boolean;
  viewportMode: 'placeholder' | 'native' | 'streamed';
  hostError?: string;
};

export type NativeViewportBounds = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type ViewportCameraInput = {
  orbitX?: number;
  orbitY?: number;
  panX?: number;
  panY?: number;
  forward?: number;
  zoom?: number;
  focusSelected?: boolean;
};

export type OpenSceneDialogOptions = {
  append?: boolean;
};

export type OpenSceneDialogResult = {
  canceled: boolean;
  filePath?: string;
  response?: unknown;
};

export type SaveSceneDialogResult = OpenSceneDialogResult;

export type HostEditTransaction = {
  id: number;
  phase: 'begin' | 'update' | 'commit' | 'cancel';
  label?: string;
};

export type ArcHostLogEvent = {
  level: 'info' | 'warning' | 'error' | 'debug';
  source: string;
  message: string;
  timestamp: string;
};

export type ArcHostEvent = {
  kind: 'event';
  sequence: number;
  type: string;
  entity: { index: number; generation: number };
  message: string;
  payload: unknown;
};

const arcApi = {
  getVersion: (): Promise<string> => ipcRenderer.invoke('app:getVersion'),
  getStartupState: (): Promise<ArcStartupState> => ipcRenderer.invoke('editor:getStartupState'),
  host: {
    query: (type: string, payload: Record<string, unknown> = {}): Promise<unknown> => ipcRenderer.invoke('host:query', type, payload),
    command: (type: string, payload: Record<string, unknown> = {}, edit?: HostEditTransaction): Promise<unknown> =>
      ipcRenderer.invoke('host:command', type, payload, edit),
    onLog: (callback: (event: ArcHostLogEvent) => void): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, logEvent: ArcHostLogEvent) => callback(logEvent);
      ipcRenderer.on('host:log', listener);
      return () => ipcRenderer.removeListener('host:log', listener);
    },
    onEvent: (callback: (event: ArcHostEvent) => void): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, hostEvent: ArcHostEvent) => callback(hostEvent);
      ipcRenderer.on('host:event', listener);
      return () => ipcRenderer.removeListener('host:event', listener);
    },
  },
  dialog: {
    openScene: (options: OpenSceneDialogOptions = {}): Promise<OpenSceneDialogResult> => ipcRenderer.invoke('dialog:openScene', options),
    saveScene: (): Promise<SaveSceneDialogResult> => ipcRenderer.invoke('dialog:saveScene'),
  },
  viewport: {
    attach: (bounds: NativeViewportBounds): Promise<unknown> => ipcRenderer.invoke('viewport:attach', bounds),
    resize: (bounds: NativeViewportBounds): Promise<unknown> => ipcRenderer.invoke('viewport:resize', bounds),
    cameraInput: (input: ViewportCameraInput): Promise<unknown> => ipcRenderer.invoke('viewport:cameraInput', input),
  },
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
