import { app, BrowserWindow, dialog, ipcMain, Menu, screen, shell } from 'electron';
import { spawn } from 'node:child_process';
import type { ChildProcessWithoutNullStreams } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { SceneGatewayCore } from './aiGatewayCore';
import { AiGatewayServer } from './aiGatewayServer';
import { ProjectService } from './projectService';
import { RecoveryService } from './recoveryService';
import { ExtensionService } from './extensionService';
import { SettingsService } from './settingsService';
import { SourceControlService } from './sourceControlService';
import type { ArcCloneProjectRequest, ArcCreateProjectRequest } from '../common/projectTypes';

const isDevelopment = !app.isPackaged;
const isCiSmoke =
  Boolean(process.env.ARC_CI_SMOKE_LOG) || process.argv.includes('--ci-smoke') || app.commandLine.hasSwitch('ci-smoke');
let ciSmokeProjectRoot: string | null = null;

declare const MAIN_WINDOW_VITE_DEV_SERVER_URL: string | undefined;
declare const MAIN_WINDOW_VITE_NAME: string;

let mainWindow: BrowserWindow | null = null;
let hostClient: ArcHostClient | null = null;
let aiGateway: AiGatewayServer | null = null;
let projectService: ProjectService | null = null;
let settingsService: SettingsService | null = null;
let sourceControlService: SourceControlService | null = null;
let recoveryService: RecoveryService | null = null;
let extensionService: ExtensionService | null = null;
let allowWindowClose = false;
let closeConfirmationPending = false;
let shutdownPending = false;
let shutdownComplete = false;

const activeWindow = (): BrowserWindow | null => (mainWindow && !mainWindow.isDestroyed() ? mainWindow : null);

const resolveProjectFile = (relativePath: string): string => {
  const project = projectService?.active();
  if (!project) throw new Error('No project is open');
  const projectRoot = fs.realpathSync(project.projectRoot);
  const normalized = relativePath.replaceAll('\\', '/').replace(/^\/+/, '');
  if (!normalized || normalized === '..' || normalized.startsWith('../') || path.isAbsolute(normalized))
    throw new Error('Project file path must be relative');
  const resolved = path.resolve(projectRoot, normalized);
  const relative = path.relative(projectRoot, resolved);
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative))
    throw new Error('Project file path escapes the project');
  const containmentTarget = fs.existsSync(resolved)
    ? fs.realpathSync(resolved)
    : fs.realpathSync(path.dirname(resolved));
  const realRelative = path.relative(projectRoot, containmentTarget);
  if (realRelative === '..' || realRelative.startsWith(`..${path.sep}`) || path.isAbsolute(realRelative))
    throw new Error('Project file path resolves outside the project');
  return resolved;
};

export type HostResponse = {
  kind: 'response';
  requestId: number;
  succeeded: boolean;
  error: string;
  payload: unknown;
  sceneRevision: number;
  worldEpoch: number;
  frameRevision: number;
};

type HostEvent = {
  kind: 'event';
  sequence: number;
  type: string;
  entity: { index: number; generation: number };
  message: string;
  payload: unknown;
};

type HostLogLevel = 'info' | 'warning' | 'error' | 'debug';

type HostLogEvent = {
  level: HostLogLevel;
  source: string;
  message: string;
  timestamp: string;
};

type NativeViewportBounds = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type CameraInput = {
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

type OpenSceneDialogOptions = {
  append?: boolean;
};

const hostLogTimestamp = () => new Date().toLocaleTimeString([], { hour12: false });

const sendHostLog = (event: Omit<HostLogEvent, 'timestamp'>): void => {
  const timestamped = {
    ...event,
    timestamp: hostLogTimestamp(),
  } satisfies HostLogEvent;
  aiGateway?.core.recordHostLog(timestamped);
  activeWindow()?.webContents.send('host:log', timestamped);
};

const normalizeHostLogLevel = (level: string): HostLogLevel => {
  const lowered = level.toLowerCase();
  if (lowered === 'warn' || lowered === 'warning') {
    return 'warning';
  }
  if (lowered === 'error' || lowered === 'fatal') {
    return 'error';
  }
  if (lowered === 'trace' || lowered === 'debug') {
    return 'debug';
  }
  return 'info';
};

const parseHostLogLine = (line: string, stream: 'stdout' | 'stderr'): Omit<HostLogEvent, 'timestamp'> => {
  const trimmed = line.trim();
  const match = trimmed.match(/^\[(trace|debug|info|warn|warning|error|fatal)\](?:\[([^\]]+)\])?\s*(.*)$/i);
  if (!match) {
    return {
      level: stream === 'stderr' ? 'error' : 'info',
      source: `host.${stream}`,
      message: trimmed,
    };
  }

  return {
    level: normalizeHostLogLevel(match[1]),
    source: match[2] || `host.${stream}`,
    message: match[3] || trimmed,
  };
};

const hostExecutableName = process.platform === 'win32' ? 'arc_host_process.exe' : 'arc_host_process';
const projectToolExecutableName = process.platform === 'win32' ? 'arc-project.exe' : 'arc-project';

const firstExistingPath = (candidates: Array<string | undefined>): string | null =>
  candidates.find((candidate): candidate is string => Boolean(candidate && fs.existsSync(candidate))) ?? null;

const resolveProjectToolPath = (): string | null => {
  const candidates: Array<string | undefined> = [
    process.env.ARC_PROJECT_TOOL_PATH,
    app.isPackaged ? path.join(process.resourcesPath, projectToolExecutableName) : undefined,
    app.isPackaged ? path.join(process.resourcesPath, 'native', projectToolExecutableName) : undefined,
  ];
  for (const root of [path.resolve(process.cwd(), '..'), path.resolve(process.cwd())]) {
    for (const preset of ['editor-vulkan', 'default', 'editor-no-vulkan']) {
      for (const configuration of ['RelWithDebInfo', 'Release', 'Debug'])
        candidates.push(path.join(root, 'out', 'build', preset, 'tools', 'project_cli', configuration, projectToolExecutableName));
      candidates.push(path.join(root, 'out', 'build', preset, 'tools', 'project_cli', projectToolExecutableName));
    }
  }
  return firstExistingPath(candidates);
};

const resolveTemplatesRoot = (): string | null =>
  firstExistingPath([
    process.env.ARC_PROJECT_TEMPLATES_PATH,
    app.isPackaged ? path.join(process.resourcesPath, 'templates') : undefined,
    app.isPackaged ? path.join(process.resourcesPath, 'share', 'arc', 'templates') : undefined,
    path.resolve(process.cwd(), '..', 'templates'),
    path.resolve(process.cwd(), 'templates'),
  ]);

const resolveHostProcessPath = (): string | null => {
  const candidates: Array<string | undefined> = [
    process.env.ARC_HOST_PROCESS_PATH,
    app.isPackaged ? path.join(process.resourcesPath, hostExecutableName) : undefined,
    app.isPackaged ? path.join(process.resourcesPath, 'native', hostExecutableName) : undefined,
  ];
  for (const root of [path.resolve(process.cwd(), '..'), path.resolve(process.cwd())]) {
    for (const preset of ['editor-vulkan', 'editor-no-vulkan']) {
      const nativeRoot = path.join(root, 'out', 'build', preset, 'editor', 'native');
      for (const configuration of ['RelWithDebInfo', 'Release', 'Debug']) {
        candidates.push(path.join(nativeRoot, configuration, hostExecutableName));
      }
      candidates.push(path.join(nativeRoot, hostExecutableName));
    }
  }

  return firstExistingPath(candidates);
};

export class ArcHostClient {
  private readonly executablePath: string | null;
  private process: ChildProcessWithoutNullStreams | null = null;
  private requestId = 1;
  private readonly pending = new Map<
    number,
    { resolve: (value: HostResponse) => void; reject: (reason: Error) => void }
  >();
  private lastError = '';
  private pendingRuntimeTick: HostEvent | null = null;
  private runtimeTickScheduled = false;
  private readonly eventListeners = new Set<(event: HostEvent) => void>();

  constructor() {
    this.executablePath = resolveHostProcessPath();
    this.start();
  }

  get connected(): boolean {
    return Boolean(this.process && !this.process.killed);
  }

  get error(): string {
    return this.lastError;
  }

  start(): void {
    if (this.process || !this.executablePath) {
      if (!this.executablePath) {
        this.lastError = 'arc_host_process was not found. Build the native editor host first.';
      }
      return;
    }

    const child = spawn(this.executablePath, [], {
      cwd: path.dirname(this.executablePath),
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    }) as ChildProcessWithoutNullStreams;
    this.process = child;

    const stdout = readline.createInterface({ input: child.stdout });
    stdout.on('line', (line) => this.handleLine(line));
    const stderr = readline.createInterface({ input: child.stderr });
    stderr.on('line', (line) => {
      this.lastError = line.trim();
      if (this.lastError) {
        sendHostLog(parseHostLogLine(this.lastError, 'stderr'));
        console.warn(`[arc_host_process] ${this.lastError}`);
      }
    });
    child.on('exit', (code, signal) => {
      const wasCurrentProcess = this.process === child;
      if (wasCurrentProcess) this.process = null;
      const exitDetail =
        this.lastError ||
        `arc_host_process exited${code === null ? '' : ` with code ${code}`}${signal ? ` (${signal})` : ''}`;
      sendHostLog({
        level: 'warning',
        source: 'host.process',
        message: exitDetail,
      });
      if (wasCurrentProcess) {
        for (const pending of this.pending.values()) {
          pending.reject(new Error(exitDetail));
        }
        this.pending.clear();
      }
    });

    if (isCiSmoke) {
      void this.command('project.open', {
        name: 'ARC CI Smoke',
        root: (() => {
          ciSmokeProjectRoot ??= fs.mkdtempSync(path.join(app.getPath('temp'), 'arc-editor-smoke-'));
          return ciSmokeProjectRoot;
        })(),
      }).catch((error) => {
        this.lastError = error instanceof Error ? error.message : String(error);
      });
    }
  }

  stop(): void {
    const child = this.process;
    this.process = null;
    child?.kill();
    const error = new Error('arc_host_process was stopped');
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }

  restart(): void {
    this.stop();
    this.lastError = '';
    this.start();
  }

  command(
    type: string,
    payload: Record<string, unknown> = {},
    edit?: Record<string, unknown>,
    expectedSceneRevision?: number,
  ): Promise<HostResponse> {
    return this.send({ kind: 'command', type, payload, edit, expectedSceneRevision });
  }

  query(type: string, payload: Record<string, unknown> = {}): Promise<HostResponse> {
    return this.send({ kind: 'query', type, payload });
  }

  onEvent(listener: (event: HostEvent) => void): () => void {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  private send(message: {
    kind: 'command' | 'query';
    type: string;
    payload: Record<string, unknown>;
    edit?: Record<string, unknown>;
    expectedSceneRevision?: number;
  }): Promise<HostResponse> {
    this.start();
    const child = this.process;
    if (!child?.stdin.writable) {
      return Promise.reject(new Error(this.lastError || 'arc_host_process is not running'));
    }

    const requestId = this.requestId++;
    const envelope = { ...message, requestId };
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      child.stdin.write(`${JSON.stringify(envelope)}\n`, (error) => {
        if (error) {
          this.pending.delete(requestId);
          reject(error);
        }
      });
    });
  }

  private handleLine(line: string): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      sendHostLog(parseHostLogLine(line, 'stdout'));
      return;
    }

    if (!parsed || typeof parsed !== 'object') {
      return;
    }

    const maybeResponse = parsed as Partial<HostResponse>;
    if ((parsed as Partial<HostEvent>).kind === 'event') {
      const event = parsed as HostEvent;
      for (const listener of this.eventListeners) listener(event);
      if (event.type === 'runtime.tickCompleted') {
        this.pendingRuntimeTick = event;
        if (!this.runtimeTickScheduled) {
          this.runtimeTickScheduled = true;
          setImmediate(() => {
            this.runtimeTickScheduled = false;
            const latest = this.pendingRuntimeTick;
            this.pendingRuntimeTick = null;
            if (latest) activeWindow()?.webContents.send('host:event', latest);
          });
        }
      } else {
        activeWindow()?.webContents.send('host:event', event);
      }
      return;
    }
    if (maybeResponse.kind !== 'response' || typeof maybeResponse.requestId !== 'number') {
      sendHostLog({
        level: 'debug',
        source: 'host.stdout',
        message: line,
      });
      return;
    }

    const pending = this.pending.get(maybeResponse.requestId);
    if (!pending) {
      return;
    }
    this.pending.delete(maybeResponse.requestId);
    pending.resolve(maybeResponse as HostResponse);
  }
}

const scaleViewportBounds = (window: BrowserWindow, bounds: NativeViewportBounds): NativeViewportBounds => {
  const display = screen.getDisplayMatching(window.getBounds());
  const scale = display.scaleFactor || 1;
  return {
    x: Math.round(bounds.x * scale),
    y: Math.round(bounds.y * scale),
    width: Math.max(1, Math.round(bounds.width * scale)),
    height: Math.max(1, Math.round(bounds.height * scale)),
  };
};

const nativeWindowHandleNumber = (window: BrowserWindow): number => {
  const handle = window.getNativeWindowHandle();
  return Number(handle.readBigUInt64LE(0));
};

type SceneDocumentState = { dirty?: boolean; sceneName?: string; activeScenePath?: string };

const saveSceneWithDialog = async (target: BrowserWindow, activeScenePath = ''): Promise<HostResponse | null> => {
  if (activeScenePath) {
    return hostClient?.command('scene.save') ?? null;
  }
  const result = await dialog.showSaveDialog(target, {
    title: 'Save ARC Scene',
    buttonLabel: 'Save',
    defaultPath: 'Untitled.arcscene',
    filters: [{ name: 'ARC Scene', extensions: ['arcscene'] }],
  });
  if (result.canceled || !result.filePath) return null;
  return hostClient?.command('scene.saveAs', { path: result.filePath }) ?? null;
};

const confirmWindowClose = async (target: BrowserWindow): Promise<void> => {
  if (closeConfirmationPending) return;
  closeConfirmationPending = true;
  try {
    const state = await hostClient?.query('scene.hierarchy');
    const document = state?.payload as SceneDocumentState | undefined;
    if (state?.succeeded && document?.dirty) {
      const choice = await dialog.showMessageBox(target, {
        type: 'warning',
        title: 'Unsaved ARC Scene',
        message: `Save changes to ${document.sceneName || 'Untitled'}?`,
        detail: 'Unsaved scene authoring changes will be lost.',
        buttons: ['Save', "Don't Save", 'Cancel'],
        defaultId: 0,
        cancelId: 2,
        noLink: true,
      });
      if (choice.response === 2) return;
      if (choice.response === 0) {
        const saved = await saveSceneWithDialog(target, document.activeScenePath);
        if (!saved?.succeeded) {
          if (saved) dialog.showErrorBox('Scene Save Failed', saved.error || 'The scene could not be saved.');
          return;
        }
      }
    }
    allowWindowClose = true;
    target.close();
  } catch (error) {
    dialog.showErrorBox('Unable to Close Scene', error instanceof Error ? error.message : String(error));
  } finally {
    closeConfirmationPending = false;
  }
};

const finishCiSmoke = (exitCode: number, message?: string): void => {
  if (message) {
    console.error(`[arc-editor-smoke] ${message}`);
  }
  const smokeLog = process.env.ARC_CI_SMOKE_LOG;
  if (smokeLog) {
    try {
      fs.writeFileSync(smokeLog, message ?? 'ok', 'utf8');
    } catch (error) {
      console.error(
        `[arc-editor-smoke] failed to write smoke diagnostic: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
  }
  hostClient?.stop();
  hostClient = null;
  if (ciSmokeProjectRoot) {
    try {
      fs.rmSync(ciSmokeProjectRoot, { recursive: true, force: true });
    } catch {
      // The host process may still be releasing its working directory on
      // Windows. The isolated directory is safe to leave for OS temp cleanup.
    }
    ciSmokeProjectRoot = null;
  }
  shutdownComplete = true;
  app.exit(exitCode);
};

const createMainWindow = (): void => {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1180,
    minHeight: 720,
    backgroundColor: '#1e1e1e',
    title: 'ARC Editor',
    autoHideMenuBar: true,
    frame: false,
    show: !isCiSmoke,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('maximize', () => mainWindow?.webContents.send('nativeWindow:maximizedChanged', true));
  mainWindow.on('unmaximize', () => mainWindow?.webContents.send('nativeWindow:maximizedChanged', false));
  mainWindow.on('close', (event) => {
    if (isCiSmoke) return;
    if (allowWindowClose) return;
    event.preventDefault();
    if (mainWindow) void confirmWindowClose(mainWindow);
  });

  if (isCiSmoke) {
    const timeout = setTimeout(
      () => finishCiSmoke(1, 'timed out waiting for renderer and native-host handshake'),
      30_000,
    );
    mainWindow.webContents.once('did-fail-load', (_event, code, description) => {
      clearTimeout(timeout);
      finishCiSmoke(1, `renderer failed to load (${code}): ${description}`);
    });
    mainWindow.webContents.once('did-finish-load', () => {
      void (async () => {
        try {
          const response = await hostClient?.query('scene.hierarchy');
          if (!response?.succeeded) {
            throw new Error(response?.error || hostClient?.error || 'native host did not answer');
          }
          clearTimeout(timeout);
          finishCiSmoke(0);
        } catch (error) {
          clearTimeout(timeout);
          finishCiSmoke(1, error instanceof Error ? error.message : String(error));
        }
      })();
    });
  }

  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    void mainWindow.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  } else {
    void mainWindow.loadFile(path.join(__dirname, `../renderer/${MAIN_WINDOW_VITE_NAME}/index.html`));
  }

  if (isDevelopment) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }
};

void app.whenReady().then(async () => {
  Menu.setApplicationMenu(null);
  hostClient = new ArcHostClient();
  projectService = new ProjectService({
    userDataPath: app.getPath('userData'),
    currentEngineVersion: app.getVersion(),
    currentEditorPath: app.getPath('exe'),
    projectToolPath: resolveProjectToolPath() ?? '',
    templatesRoot: resolveTemplatesRoot() ?? '',
    host: hostClient,
  });
  settingsService = new SettingsService(
    path.join(app.getPath('userData'), 'editor-settings.v1.json'),
    () => projectService?.active() ?? null,
  );
  sourceControlService = new SourceControlService(
    () => projectService?.active()?.projectRoot ?? null,
    () => settingsService?.snapshot().values['sourceControl.provider'] !== 'none',
    () => projectService?.active()?.writable ?? false,
  );
  extensionService = new ExtensionService(
    () => projectService?.active() ?? null,
    app.getVersion(),
    () => settingsService?.snapshot().values['extensions.allowProjectExtensions'] !== false,
  );
  recoveryService = new RecoveryService(
    path.join(app.getPath('userData'), 'bootstrap-recovery'),
    hostClient,
    () => settingsService?.snapshot().values ?? {},
  );
  hostClient.onEvent((event) => {
    if (
      event.type === 'component.changed' ||
      event.type === 'scene.changed' ||
      event.type === 'hierarchy.changed' ||
      event.type === 'terrain.strokeCommitted'
    )
      recoveryService?.noteMutation();
  });
  const suppliedProject = process.argv.find((argument) => argument.toLowerCase().endsWith('.arcproject'));
  if (suppliedProject) {
    const opened = await projectService.open(suppliedProject);
    if (opened.succeeded && opened.project) recoveryService.start(opened.project);
    else
      sendHostLog({
        level: 'error',
        source: 'project.browser',
        message: opened.error || 'The supplied project could not be opened',
      });
  }
  if (!isCiSmoke) {
    const gatewayCore = new SceneGatewayCore(hostClient);
    hostClient.onEvent((event) => {
      gatewayCore.recordHostEvent(event);
      if (
        event.type === 'project.opened' ||
        event.type === 'project.closed' ||
        (event.type === 'scene.changed' && /opened|loaded|new scene/i.test(event.message))
      ) {
        void gatewayCore.invalidateAuthority(event.message || event.type);
      }
    });
    aiGateway = new AiGatewayServer(gatewayCore, {
      appDataPath: app.getPath('userData'),
      onStatus: (status) => activeWindow()?.webContents.send('ai-gateway:status', status),
    });
    try {
      await aiGateway.start();
    } catch (error) {
      sendHostLog({
        level: 'error',
        source: 'ai.gateway',
        message: `AI gateway failed to start: ${error instanceof Error ? error.message : String(error)}`,
      });
    }
  }

  ipcMain.handle('app:getVersion', () => app.getVersion());
  ipcMain.handle('editor:getStartupState', () => ({
    appVersion: app.getVersion(),
    engineHostConnected: hostClient?.connected ?? false,
    viewportMode: hostClient?.connected ? 'native' : 'unavailable',
    hostError: hostClient?.error ?? '',
    activeProject: projectService?.snapshot().activeProject ?? null,
    ciSmoke: isCiSmoke,
  }));
  ipcMain.handle('project:snapshot', () => projectService?.snapshot() ?? null);
  ipcMain.handle(
    'project:open',
    async (_event, candidate: string, options: { readOnly?: boolean; upgrade?: boolean } = {}) => {
      const result = await projectService?.open(candidate, options);
      if (result?.succeeded && result.project) {
        recoveryService?.start(result.project);
        extensionService?.invalidate();
      }
      return result;
    },
  );
  ipcMain.handle('project:close', async () => {
    const target = activeWindow();
    let scene: SceneDocumentState | undefined;
    try {
      scene = (await hostClient?.query('scene.hierarchy'))?.payload as SceneDocumentState | undefined;
    } catch {
      // A crashed host cannot report dirty state; closing the project remains available.
    }
    if (target && scene?.dirty) {
      const choice = await dialog.showMessageBox(target, {
        type: 'warning',
        title: 'Close ARC Project',
        message: `Save changes to ${scene.sceneName || 'Untitled'} before closing the project?`,
        detail: 'Unsaved scene authoring changes will be lost.',
        buttons: ['Save', "Don't Save", 'Cancel'],
        defaultId: 0,
        cancelId: 2,
        noLink: true,
      });
      if (choice.response === 2) return { succeeded: false, error: 'Project close cancelled' };
      if (choice.response === 0) {
        const saved = await saveSceneWithDialog(target, scene.activeScenePath);
        if (!saved?.succeeded) return { succeeded: false, error: saved?.error || 'Scene save failed' };
      }
    }
    const result = await projectService?.close();
    if (result?.succeeded) recoveryService?.stop(true);
    return result;
  });
  ipcMain.handle('project:create', (_event, request: ArcCreateProjectRequest) => projectService?.create(request));
  ipcMain.handle('project:launchMatchingEngine', (_event, candidate: string) =>
    projectService?.launchMatchingEngine(candidate),
  );
  ipcMain.handle('project:clone', (_event, request: ArcCloneProjectRequest) => projectService?.clone(request));
  ipcMain.handle('project:removeRecent', (_event, descriptorPath: string) =>
    projectService?.removeRecent(descriptorPath),
  );
  ipcMain.handle('project:readText', (_event, relativePath: string) => {
    const target = resolveProjectFile(relativePath);
    const stats = fs.statSync(target);
    if (!stats.isFile() || stats.size > 8 * 1024 * 1024)
      throw new Error('Project text file is unavailable or too large');
    return {
      path: relativePath.replaceAll('\\', '/'),
      text: fs.readFileSync(target, 'utf8'),
      modifiedAt: stats.mtime.toISOString(),
    };
  });
  ipcMain.handle('project:writeText', (_event, relativePath: string, text: string) => {
    if (!projectService?.active()?.writable) throw new Error('The active project is read-only');
    if (Buffer.byteLength(text, 'utf8') > 8 * 1024 * 1024) throw new Error('Project text file is too large');
    const target = resolveProjectFile(relativePath);
    const temporary = `${target}.tmp-${process.pid}`;
    fs.writeFileSync(temporary, text, 'utf8');
    fs.renameSync(temporary, target);
    return { succeeded: true };
  });
  ipcMain.handle('settings:snapshot', () => settingsService?.snapshot() ?? null);
  ipcMain.handle(
    'settings:update',
    (_event, scope: 'user' | 'project', changes: Record<string, unknown>, expectedRevision: number) => {
      const updated = settingsService?.update(scope, changes, expectedRevision);
      if (Object.hasOwn(changes, 'extensions.allowProjectExtensions')) extensionService?.invalidate();
      return updated;
    },
  );
  ipcMain.handle('vcs:snapshot', () => sourceControlService?.snapshot());
  ipcMain.handle('vcs:diff', (_event, filePath: string, staged = false) =>
    sourceControlService?.diff(filePath, staged),
  );
  ipcMain.handle('vcs:stage', (_event, paths: string[]) => sourceControlService?.stage(paths));
  ipcMain.handle('vcs:unstage', (_event, paths: string[]) => sourceControlService?.unstage(paths));
  ipcMain.handle('vcs:discard', async (_event, paths: string[]) => {
    const target = activeWindow();
    if (!target) throw new Error('No active editor window');
    const confirmation = await dialog.showMessageBox(target, {
      type: 'warning',
      title: 'Discard local changes?',
      message: `Discard changes in ${paths.length} file(s)?`,
      detail: 'This operation cannot be undone by ARC.',
      buttons: ['Discard', 'Cancel'],
      defaultId: 1,
      cancelId: 1,
      noLink: true,
    });
    return confirmation.response === 0
      ? sourceControlService?.discard(paths)
      : { succeeded: false, output: '', error: 'Discard cancelled' };
  });
  ipcMain.handle('vcs:checkout', async (_event, reference: string) => {
    const target = activeWindow();
    if (!target) throw new Error('No active editor window');
    const confirmation = await dialog.showMessageBox(target, {
      type: 'question',
      title: 'Switch Git reference?',
      message: `Checkout '${reference}'?`,
      detail: 'Unsaved working tree changes may prevent or complicate the checkout.',
      buttons: ['Checkout', 'Cancel'],
      defaultId: 1,
      cancelId: 1,
      noLink: true,
    });
    return confirmation.response === 0
      ? sourceControlService?.checkout(reference)
      : { succeeded: false, output: '', error: 'Checkout cancelled' };
  });
  ipcMain.handle('vcs:pull', () => sourceControlService?.pull());
  ipcMain.handle('vcs:push', () => sourceControlService?.push());
  ipcMain.handle('vcs:commit', (_event, message: string) => sourceControlService?.commit(message));
  ipcMain.handle('recovery:snapshot', (_event, projectGuid?: string, projectRoot?: string) => recoveryService?.snapshot(projectGuid, projectRoot) ?? null);
  ipcMain.handle('recovery:restore', (_event, id: string) => recoveryService?.restore(id));
  ipcMain.handle('recovery:discard', (_event, id: string) => recoveryService?.discard(id) ?? false);
  ipcMain.handle('extensions:snapshot', (_event, force = false) => extensionService?.snapshot(force) ?? null);
  ipcMain.handle('dialog:openProject', async () => {
    const target = activeWindow();
    if (!target) throw new Error('No active editor window');
    const result = await dialog.showOpenDialog(target, {
      title: 'Open ARC Project',
      buttonLabel: 'Open Project',
      properties: ['openFile'],
      filters: [{ name: 'ARC Project', extensions: ['arcproject'] }],
    });
    return result.canceled || !result.filePaths.length ? null : result.filePaths[0];
  });
  ipcMain.handle('dialog:projectDestination', async (_event, title = 'Select Project Destination') => {
    const target = activeWindow();
    if (!target) throw new Error('No active editor window');
    const result = await dialog.showOpenDialog(target, {
      title,
      buttonLabel: 'Select Folder',
      properties: ['openDirectory', 'createDirectory'],
    });
    return result.canceled || !result.filePaths.length ? null : result.filePaths[0];
  });

  ipcMain.handle('host:query', (_event, type: string, payload: Record<string, unknown> = {}) =>
    hostClient?.query(type, payload),
  );
  ipcMain.handle('host:reconnect', async () => {
    hostClient?.restart();
    const project = projectService?.active() ?? null;
    if (!hostClient?.connected) {
      return {
        appVersion: app.getVersion(),
        engineHostConnected: false,
        viewportMode: 'unavailable',
        hostError: hostClient?.error || 'Native editor host is unavailable',
        activeProject: project,
      };
    }
    if (project) {
      const reopened = await projectService?.open(project.descriptorPath, { readOnly: !project.writable });
      if (!reopened?.succeeded) {
        return {
          appVersion: app.getVersion(),
          engineHostConnected: false,
          viewportMode: 'unavailable',
          hostError: reopened?.error || 'Native host could not reopen the active project',
          activeProject: project,
        };
      }
    }
    return {
      appVersion: app.getVersion(),
      engineHostConnected: true,
      viewportMode: 'native',
      hostError: '',
      activeProject: project,
    };
  });
  ipcMain.handle(
    'host:command',
    (_event, type: string, payload: Record<string, unknown>, edit?: Record<string, unknown>) =>
      hostClient?.command(type, payload, edit),
  );
  ipcMain.handle('ai-gateway:status', () => aiGateway?.core.status() ?? null);
  ipcMain.handle('ai-gateway:approve', (_event, requestId: string) => aiGateway?.core.approveEdit(requestId) ?? false);
  ipcMain.handle('ai-gateway:deny', (_event, requestId: string) => aiGateway?.core.denyEdit(requestId) ?? false);
  ipcMain.handle('ai-gateway:revoke', async (_event, clientId: string) => {
    await aiGateway?.core.revokeClient(clientId);
  });
  ipcMain.handle('ai-gateway:cancelEdit', async (_event, sessionId: string, clientId: string) =>
    aiGateway?.core.invoke('edit.cancel', { editSessionId: sessionId }, clientId),
  );
  ipcMain.handle('ai-gateway:undoLastEdit', async () => aiGateway?.core.undoLastCommittedEdit());
  ipcMain.handle('dialog:openScene', async (_event, options: OpenSceneDialogOptions = {}) => {
    const target = activeWindow();
    if (!target) {
      throw new Error('No active editor window');
    }
    const result = await dialog.showOpenDialog(target, {
      title: options.append ? 'Import Scene Into Current' : 'Open Scene',
      buttonLabel: options.append ? 'Import' : 'Open',
      properties: ['openFile'],
      filters: [
        { name: 'Scene Assets', extensions: ['arcscene', 'glb', 'gltf', 'fbx', 'scene'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    if (result.canceled || result.filePaths.length === 0) {
      return { canceled: true };
    }

    const filePath = result.filePaths[0];
    const response = await hostClient?.command('scene.open', {
      path: filePath,
      append: Boolean(options.append),
    });
    return {
      canceled: false,
      filePath,
      response,
    };
  });
  ipcMain.handle('dialog:saveScene', async () => {
    const target = activeWindow();
    if (!target) throw new Error('No active editor window');
    const result = await dialog.showSaveDialog(target, {
      title: 'Save ARC Scene',
      buttonLabel: 'Save',
      defaultPath: 'Untitled.arcscene',
      filters: [{ name: 'ARC Scene', extensions: ['arcscene'] }],
    });
    if (result.canceled || !result.filePath) return { canceled: true };
    const response = await hostClient?.command('scene.saveAs', { path: result.filePath });
    return { canceled: false, filePath: result.filePath, response };
  });
  ipcMain.handle('dialog:createPrefab', async (_event, entity: { index: number; generation: number }) => {
    const target = activeWindow();
    if (!target) throw new Error('No active editor window');
    const result = await dialog.showSaveDialog(target, {
      title: 'Create ARC Prefab',
      buttonLabel: 'Create Prefab',
      defaultPath: 'NewPrefab.arcprefab',
      filters: [{ name: 'ARC Prefab', extensions: ['arcprefab'] }],
    });
    if (result.canceled || !result.filePath) return { canceled: true };
    const response = await hostClient?.command('prefab.create', { entity, path: result.filePath });
    return { canceled: false, filePath: result.filePath, response };
  });
  ipcMain.handle('dialog:instantiatePrefab', async (_event, parent?: { index: number; generation: number }) => {
    const target = activeWindow();
    if (!target) throw new Error('No active editor window');
    const result = await dialog.showOpenDialog(target, {
      title: 'Instantiate ARC Prefab',
      buttonLabel: 'Instantiate',
      properties: ['openFile'],
      filters: [{ name: 'ARC Prefab', extensions: ['arcprefab'] }],
    });
    if (result.canceled || result.filePaths.length === 0) return { canceled: true };
    const filePath = result.filePaths[0];
    const response = await hostClient?.command('prefab.instantiate', {
      path: filePath,
      ...(parent ? { parent } : {}),
    });
    return { canceled: false, filePath, response };
  });
  ipcMain.handle('viewport:attach', (_event, bounds: NativeViewportBounds) => {
    if (isCiSmoke) return { skipped: true, reason: 'ci-smoke' };
    const target = activeWindow();
    if (!target) {
      throw new Error('No active editor window');
    }
    const scaled = scaleViewportBounds(target, bounds);
    return hostClient?.command('viewport.attach', {
      nativeHandle: nativeWindowHandleNumber(target),
      ...scaled,
    });
  });
  ipcMain.handle('viewport:resize', (_event, bounds: NativeViewportBounds) => {
    if (isCiSmoke) return { skipped: true, reason: 'ci-smoke' };
    const target = activeWindow();
    if (!target) {
      throw new Error('No active editor window');
    }
    return hostClient?.command('viewport.resize', scaleViewportBounds(target, bounds));
  });
  ipcMain.handle('viewport:cameraInput', (_event, input: CameraInput) =>
    hostClient?.command('viewport.cameraInput', input),
  );

  ipcMain.handle('nativeWindow:minimize', () => activeWindow()?.minimize());
  ipcMain.handle('nativeWindow:toggleMaximize', () => {
    const target = activeWindow();
    if (!target) {
      return false;
    }

    if (target.isMaximized()) {
      target.unmaximize();
      return false;
    }

    target.maximize();
    return true;
  });
  ipcMain.handle('nativeWindow:close', () => activeWindow()?.close());
  ipcMain.handle('nativeWindow:isMaximized', () => activeWindow()?.isMaximized() ?? false);

  createMainWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (isCiSmoke) return;
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', (event) => {
  if (shutdownComplete) return;
  event.preventDefault();
  if (shutdownPending) return;
  shutdownPending = true;
  void (async () => {
    try {
      await aiGateway?.stop();
    } finally {
      aiGateway = null;
      projectService = null;
      settingsService = null;
      sourceControlService = null;
      recoveryService?.stop(true);
      recoveryService = null;
      extensionService = null;
      hostClient?.stop();
      hostClient = null;
      shutdownComplete = true;
      app.quit();
    }
  })();
});
