import { contextBridge, ipcRenderer, sharedTexture } from 'electron';
import type {
  ArcCloneProjectRequest,
  ArcCreateProjectRequest,
  ArcProjectBrowserSnapshot,
  ArcProjectCandidate,
  ArcProjectOperationResult,
} from '../common/projectTypes';
import type {
  EditorSettingsSnapshot,
  ProjectTextFile,
  RecoverySnapshot,
  SourceControlResult,
  SourceControlSnapshot,
} from '../common/editorWorkflowTypes';
import type { ArcExtensionSnapshot } from '../common/extensionTypes';
import type { ArcBuildRequest, ArcBuildSnapshot } from '../common/buildTypes';

export type ArcStartupState = {
  appVersion: string;
  engineHostConnected: boolean;
  viewportMode: 'unavailable' | 'native' | 'streamed';
  hostError?: string;
  activeProject?: ArcProjectCandidate | null;
};

export type NativeViewportBounds = {
  viewportId: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

type SharedViewportMetadata = {
  viewportId: string;
  frameId: number;
  generation: number;
  width: number;
  height: number;
};

const sharedViewportSurfaces = new Map<string, string>();
sharedTexture.setSharedTextureReceiver(async ({ importedSharedTexture }, metadata: SharedViewportMetadata) => {
  try {
    const elementId = sharedViewportSurfaces.get(metadata.viewportId);
    const canvas = elementId ? document.getElementById(elementId) : null;
    if (!(canvas instanceof HTMLCanvasElement)) return;
    if (canvas.width !== metadata.width) canvas.width = metadata.width;
    if (canvas.height !== metadata.height) canvas.height = metadata.height;
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) return;
    const frame = importedSharedTexture.getVideoFrame();
    try {
      context.drawImage(frame, 0, 0, canvas.width, canvas.height);
    } finally {
      frame.close();
    }
  } finally {
    importedSharedTexture.release();
  }
});

export type ViewportCameraInput = {
  viewportId?: string;
  orbitX?: number;
  orbitY?: number;
  lookX?: number;
  lookY?: number;
  panX?: number;
  panY?: number;
  forward?: number;
  zoom?: number;
  focusSelected?: boolean;
};

export type ViewportPointerInput = {
  viewportId: string;
  phase: 'down' | 'move' | 'up' | 'wheel' | 'leave' | 'cancel';
  x: number;
  y: number;
  button?: number;
  wheel?: number;
  alt?: boolean;
  shift?: boolean;
  control?: boolean;
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
    id: string;
    clientId: string;
    clientName: string;
    label: string;
    requestedAt: string;
    state: 'pending' | 'approved' | 'denied' | 'expired';
    expiresAt?: string;
  }>;
  activeEditSession: {
    id: string;
    clientId: string;
    label: string;
    startedAt: string;
    lastActivityAt: string;
    expectedSceneRevision: number;
  } | null;
  lastCommittedEdit: {
    clientId: string;
    label: string;
    sceneRevision: number;
    committedAt: string;
  } | null;
  viewportLease: { clientId: string; expiresAt: string } | null;
  audit: Array<{
    sequence: number;
    timestamp: string;
    clientId: string;
    category: string;
    operation: string;
    succeeded: boolean;
    detail: string;
  }>;
};

const arcApi = {
  getVersion: (): Promise<string> => ipcRenderer.invoke('app:getVersion'),
  getStartupState: (): Promise<ArcStartupState> => ipcRenderer.invoke('editor:getStartupState'),
  projects: {
    snapshot: (): Promise<ArcProjectBrowserSnapshot | null> => ipcRenderer.invoke('project:snapshot'),
    open: (
      candidate: string,
      options: { readOnly?: boolean; upgrade?: boolean } = {},
    ): Promise<ArcProjectOperationResult> => ipcRenderer.invoke('project:open', candidate, options),
    close: (): Promise<ArcProjectOperationResult> => ipcRenderer.invoke('project:close'),
    launchMatchingEngine: (candidate: string): Promise<ArcProjectOperationResult> =>
      ipcRenderer.invoke('project:launchMatchingEngine', candidate),
    create: (request: ArcCreateProjectRequest): Promise<ArcProjectOperationResult> =>
      ipcRenderer.invoke('project:create', request),
    clone: (request: ArcCloneProjectRequest): Promise<ArcProjectOperationResult> =>
      ipcRenderer.invoke('project:clone', request),
    removeRecent: (descriptorPath: string): Promise<void> => ipcRenderer.invoke('project:removeRecent', descriptorPath),
    delete: (descriptorPath: string): Promise<ArcProjectOperationResult> =>
      ipcRenderer.invoke('project:delete', descriptorPath),
    readText: (path: string): Promise<ProjectTextFile> => ipcRenderer.invoke('project:readText', path),
    writeText: (path: string, text: string): Promise<{ succeeded: boolean }> =>
      ipcRenderer.invoke('project:writeText', path, text),
  },
  settings: {
    snapshot: (): Promise<EditorSettingsSnapshot | null> => ipcRenderer.invoke('settings:snapshot'),
    update: (
      scope: 'user' | 'project',
      changes: Record<string, unknown>,
      expectedRevision: number,
    ): Promise<EditorSettingsSnapshot | null> =>
      ipcRenderer.invoke('settings:update', scope, changes, expectedRevision),
  },
  sourceControl: {
    snapshot: (): Promise<SourceControlSnapshot | undefined> => ipcRenderer.invoke('vcs:snapshot'),
    diff: (path: string, staged = false): Promise<SourceControlResult | undefined> =>
      ipcRenderer.invoke('vcs:diff', path, staged),
    stage: (paths: string[]): Promise<SourceControlResult | undefined> => ipcRenderer.invoke('vcs:stage', paths),
    unstage: (paths: string[]): Promise<SourceControlResult | undefined> => ipcRenderer.invoke('vcs:unstage', paths),
    discard: (paths: string[]): Promise<SourceControlResult | undefined> => ipcRenderer.invoke('vcs:discard', paths),
    checkout: (reference: string): Promise<SourceControlResult | undefined> =>
      ipcRenderer.invoke('vcs:checkout', reference),
    pull: (): Promise<SourceControlResult | undefined> => ipcRenderer.invoke('vcs:pull'),
    push: (): Promise<SourceControlResult | undefined> => ipcRenderer.invoke('vcs:push'),
    commit: (message: string): Promise<SourceControlResult | undefined> => ipcRenderer.invoke('vcs:commit', message),
  },
  recovery: {
    snapshot: (projectGuid?: string, projectRoot?: string): Promise<RecoverySnapshot | null> =>
      ipcRenderer.invoke('recovery:snapshot', projectGuid, projectRoot),
    restore: (id: string): Promise<unknown> => ipcRenderer.invoke('recovery:restore', id),
    discard: (id: string): Promise<boolean> => ipcRenderer.invoke('recovery:discard', id),
  },
  extensions: {
    snapshot: (force = false): Promise<ArcExtensionSnapshot | null> => ipcRenderer.invoke('extensions:snapshot', force),
    executeCommand: (id: string, arguments_: unknown[] = []): Promise<unknown> =>
      ipcRenderer.invoke('extensions:executeCommand', id, arguments_),
  },
  build: {
    snapshot: (): Promise<ArcBuildSnapshot | null> => ipcRenderer.invoke('build:snapshot'),
    execute: (request: ArcBuildRequest): Promise<ArcBuildSnapshot | null> =>
      ipcRenderer.invoke('build:execute', request),
    openDiagnostic: (file: string, line?: number, column?: number): Promise<string> =>
      ipcRenderer.invoke('build:openDiagnostic', file, line, column),
    onState: (callback: (snapshot: ArcBuildSnapshot) => void): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, snapshot: ArcBuildSnapshot) => callback(snapshot);
      ipcRenderer.on('build:state', listener);
      return () => ipcRenderer.removeListener('build:state', listener);
    },
  },
  host: {
    reconnect: (): Promise<ArcStartupState> => ipcRenderer.invoke('host:reconnect'),
    query: (type: string, payload: Record<string, unknown> = {}): Promise<unknown> =>
      ipcRenderer.invoke('host:query', type, payload),
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
    openProject: (): Promise<string | null> => ipcRenderer.invoke('dialog:openProject'),
    projectDestination: (title?: string): Promise<string | null> =>
      ipcRenderer.invoke('dialog:projectDestination', title),
    openScene: (options: OpenSceneDialogOptions = {}): Promise<OpenSceneDialogResult> =>
      ipcRenderer.invoke('dialog:openScene', options),
    saveScene: (): Promise<SaveSceneDialogResult> => ipcRenderer.invoke('dialog:saveScene'),
    createPrefab: (entity: { index: number; generation: number }): Promise<PrefabDialogResult> =>
      ipcRenderer.invoke('dialog:createPrefab', entity),
    instantiatePrefab: (parent?: { index: number; generation: number }): Promise<PrefabDialogResult> =>
      ipcRenderer.invoke('dialog:instantiatePrefab', parent),
  },
  viewport: {
    create: (bounds: NativeViewportBounds): Promise<unknown> => ipcRenderer.invoke('viewport:create', bounds),
    attach: (bounds: NativeViewportBounds): Promise<unknown> => ipcRenderer.invoke('viewport:attach', bounds),
    resize: (bounds: NativeViewportBounds): Promise<unknown> => ipcRenderer.invoke('viewport:resize', bounds),
    detach: (viewportId: string): Promise<unknown> => ipcRenderer.invoke('viewport:detach', viewportId),
    cameraInput: (input: ViewportCameraInput): Promise<unknown> => ipcRenderer.invoke('viewport:cameraInput', input),
    setVisibility: (viewportId: string, visible: boolean): Promise<unknown> =>
      ipcRenderer.invoke('viewport:setVisibility', viewportId, visible),
    pointer: (input: ViewportPointerInput): Promise<unknown> => ipcRenderer.invoke('viewport:pointer', input),
    key: (input: Record<string, unknown>): Promise<unknown> => ipcRenderer.invoke('viewport:key', input),
    registerSurface: (viewportId: string, elementId: string): void => {
      sharedViewportSurfaces.set(viewportId, elementId);
    },
    unregisterSurface: (viewportId: string): void => {
      sharedViewportSurfaces.delete(viewportId);
    },
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
