import { app, BrowserWindow, dialog, ipcMain, Menu, screen, shell } from 'electron';
import { spawn } from 'node:child_process';
import type { ChildProcessWithoutNullStreams } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';

const isDevelopment = !app.isPackaged;

declare const MAIN_WINDOW_VITE_DEV_SERVER_URL: string | undefined;
declare const MAIN_WINDOW_VITE_NAME: string;

let mainWindow: BrowserWindow | null = null;
let hostClient: ArcHostClient | null = null;

const activeWindow = (): BrowserWindow | null => mainWindow && !mainWindow.isDestroyed() ? mainWindow : null;

type HostResponse = {
  kind: 'response';
  requestId: number;
  succeeded: boolean;
  error: string;
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
  activeWindow()?.webContents.send('host:log', {
    ...event,
    timestamp: hostLogTimestamp(),
  } satisfies HostLogEvent);
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

const resolveHostProcessPath = (): string | null => {
  const candidates = [
    process.env.ARC_HOST_PROCESS_PATH,
    path.resolve(process.cwd(), '..', 'out', 'build', 'editor-vulkan', 'editor', 'Release', hostExecutableName),
    path.resolve(process.cwd(), '..', 'out', 'build', 'editor-vulkan', 'editor', 'Debug', hostExecutableName),
    path.resolve(process.cwd(), '..', 'out', 'build', 'editor-vulkan', 'editor', hostExecutableName),
    path.resolve(process.cwd(), '..', 'out', 'build', 'editor-no-vulkan', 'editor', 'Release', hostExecutableName),
    path.resolve(process.cwd(), '..', 'out', 'build', 'editor-no-vulkan', 'editor', 'Debug', hostExecutableName),
    path.resolve(process.cwd(), '..', 'out', 'build', 'editor-no-vulkan', 'editor', hostExecutableName),
    path.resolve(process.cwd(), 'out', 'build', 'editor-vulkan', 'editor', 'Release', hostExecutableName),
    path.resolve(process.cwd(), 'out', 'build', 'editor-vulkan', 'editor', 'Debug', hostExecutableName),
    path.resolve(process.cwd(), 'out', 'build', 'editor-vulkan', 'editor', hostExecutableName),
    path.resolve(process.cwd(), 'out', 'build', 'editor-no-vulkan', 'editor', 'Release', hostExecutableName),
    path.resolve(process.cwd(), 'out', 'build', 'editor-no-vulkan', 'editor', 'Debug', hostExecutableName),
    path.resolve(process.cwd(), 'out', 'build', 'editor-no-vulkan', 'editor', hostExecutableName),
    path.resolve(process.cwd(), '..', 'build', 'editor', 'Release', hostExecutableName),
    path.resolve(process.cwd(), '..', 'build', 'editor', 'Debug', hostExecutableName),
    path.resolve(process.cwd(), '..', 'build', 'editor', hostExecutableName),
    path.resolve(process.cwd(), 'build', 'editor', 'Release', hostExecutableName),
    path.resolve(process.cwd(), 'build', 'editor', 'Debug', hostExecutableName),
    path.resolve(process.cwd(), 'build', 'editor', hostExecutableName),
  ].filter((candidate): candidate is string => Boolean(candidate));

  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null;
};

class ArcHostClient {
  private readonly executablePath: string | null;
  private process: ChildProcessWithoutNullStreams | null = null;
  private requestId = 1;
  private readonly pending = new Map<number, { resolve: (value: HostResponse) => void; reject: (reason: Error) => void }>();
  private lastError = '';

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
        this.lastError = 'arc_host_process was not found. Build the editor target first.';
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
    child.on('exit', () => {
      this.process = null;
      sendHostLog({
        level: 'warning',
        source: 'host.process',
        message: 'arc_host_process exited',
      });
      for (const pending of this.pending.values()) {
        pending.reject(new Error('arc_host_process exited'));
      }
      this.pending.clear();
    });

    void this.command('project.open', {
      name: 'Arc Sandbox',
      root: path.resolve(process.cwd(), '..'),
    }).catch((error) => {
      this.lastError = error instanceof Error ? error.message : String(error);
    });
  }

  stop(): void {
    this.process?.kill();
    this.process = null;
  }

  command(type: string, payload: Record<string, unknown> = {}): Promise<HostResponse> {
    return this.send({ kind: 'command', type, payload });
  }

  query(type: string, payload: Record<string, unknown> = {}): Promise<HostResponse> {
    return this.send({ kind: 'query', type, payload });
  }

  private send(message: { kind: 'command' | 'query'; type: string; payload: Record<string, unknown> }): Promise<HostResponse> {
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

const createMainWindow = (): void => {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1180,
    minHeight: 720,
    backgroundColor: '#1e1e1e',
    title: 'arc editor2',
    autoHideMenuBar: true,
    frame: false,
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

  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    void mainWindow.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  } else {
    void mainWindow.loadFile(path.join(__dirname, `../renderer/${MAIN_WINDOW_VITE_NAME}/index.html`));
  }

  if (isDevelopment) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }
};

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);
  hostClient = new ArcHostClient();

  ipcMain.handle('app:getVersion', () => app.getVersion());
  ipcMain.handle('editor:getStartupState', () => ({
    appVersion: app.getVersion(),
    engineHostConnected: hostClient?.connected ?? false,
    viewportMode: hostClient?.connected ? 'native' : 'placeholder',
    hostError: hostClient?.error ?? '',
  }));

  ipcMain.handle('host:query', (_event, type: string, payload: Record<string, unknown> = {}) => hostClient?.query(type, payload));
  ipcMain.handle('host:command', (_event, type: string, payload: Record<string, unknown>) => hostClient?.command(type, payload));
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
        { name: 'Scene Assets', extensions: ['glb', 'gltf', 'fbx', 'scene'] },
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
  ipcMain.handle('viewport:attach', (_event, bounds: NativeViewportBounds) => {
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
    const target = activeWindow();
    if (!target) {
      throw new Error('No active editor window');
    }
    return hostClient?.command('viewport.resize', scaleViewportBounds(target, bounds));
  });
  ipcMain.handle('viewport:cameraInput', (_event, input: CameraInput) => hostClient?.command('viewport.cameraInput', input));

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
  hostClient?.stop();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
