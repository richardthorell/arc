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
export type PrefabDialogResult = OpenSceneDialogResult;

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

export type ArcAiGatewayStatus = {
  enabled: boolean;
  endpoint: string;
  discoveryFile: string;
  protocolVersion: number;
  sceneRevision: number;
  worldEpoch: number;
  frameRevision: number;
  eventSequence: number;
  clients: Array<{ id: string; name: string; connectedAt: string; lastSeenAt: string }>;
  pendingEditRequests: Array<{
    id: string; clientId: string; clientName: string; label: string; requestedAt: string;
    state: 'pending' | 'approved' | 'denied' | 'expired'; expiresAt?: string;
  }>;
  activeEditSession: {
    id: string; clientId: string; label: string; startedAt: string; lastActivityAt: string;
    expectedSceneRevision: number;
  } | null;
  lastCommittedEdit: {
    clientId: string; label: string; sceneRevision: number; committedAt: string;
  } | null;
  viewportLease: { clientId: string; expiresAt: string } | null;
  audit: Array<{
    sequence: number; timestamp: string; clientId: string; category: string;
    operation: string; succeeded: boolean; detail: string;
  }>;
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
  aiGateway: {
    status: (): Promise<ArcAiGatewayStatus | null> => ipcRenderer.invoke('ai-gateway:status'),
    approve: (requestId: string): Promise<boolean> => ipcRenderer.invoke('ai-gateway:approve', requestId),
    deny: (requestId: string): Promise<boolean> => ipcRenderer.invoke('ai-gateway:deny', requestId),
    revoke: (clientId: string): Promise<void> => ipcRenderer.invoke('ai-gateway:revoke', clientId),
    cancelEdit: (sessionId: string, clientId: string): Promise<unknown> =>
      ipcRenderer.invoke('ai-gateway:cancelEdit', sessionId, clientId),
    undoLastEdit: (): Promise<unknown> => ipcRenderer.invoke('ai-gateway:undoLastEdit'),
    onStatus: (callback: (status: ArcAiGatewayStatus) => void): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, status: ArcAiGatewayStatus) => callback(status);
      ipcRenderer.on('ai-gateway:status', listener);
      return () => ipcRenderer.removeListener('ai-gateway:status', listener);
    },
  },
  dialog: {
    openScene: (options: OpenSceneDialogOptions = {}): Promise<OpenSceneDialogResult> => ipcRenderer.invoke('dialog:openScene', options),
    saveScene: (): Promise<SaveSceneDialogResult> => ipcRenderer.invoke('dialog:saveScene'),
    createPrefab: (entity: { index: number; generation: number }): Promise<PrefabDialogResult> =>
      ipcRenderer.invoke('dialog:createPrefab', entity),
    instantiatePrefab: (parent?: { index: number; generation: number }): Promise<PrefabDialogResult> =>
      ipcRenderer.invoke('dialog:instantiatePrefab', parent),
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
