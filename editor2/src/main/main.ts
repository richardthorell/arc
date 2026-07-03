import { app, BrowserWindow, ipcMain, Menu, shell } from 'electron';
import path from 'node:path';

const isDevelopment = !app.isPackaged;

declare const MAIN_WINDOW_VITE_DEV_SERVER_URL: string | undefined;
declare const MAIN_WINDOW_VITE_NAME: string;

let mainWindow: BrowserWindow | null = null;

const activeWindow = (): BrowserWindow | null => mainWindow && !mainWindow.isDestroyed() ? mainWindow : null;

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

  ipcMain.handle('app:getVersion', () => app.getVersion());
  ipcMain.handle('editor:getStartupState', () => ({
    appVersion: app.getVersion(),
    engineHostConnected: false,
    viewportMode: 'placeholder',
  }));

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
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
