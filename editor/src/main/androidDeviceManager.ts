import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { dialog } from 'electron';

export const androidStudioCandidates = (platform: NodeJS.Platform, env: NodeJS.ProcessEnv, home: string): string[] => {
  if (platform === 'win32') {
    return [env.ProgramFiles, env['ProgramFiles(x86)'], env.LOCALAPPDATA]
      .filter((root): root is string => Boolean(root))
      .map((root) => path.join(root, 'Android', 'Android Studio', 'bin', 'studio64.exe'));
  }
  if (platform === 'darwin') {
    return ['/Applications/Android Studio.app', path.join(home, 'Applications', 'Android Studio.app')];
  }
  return ['/opt/android-studio/bin/studio.sh', '/usr/local/android-studio/bin/studio.sh'];
};

let pending: Promise<void> | undefined;

// Device Manager is an Android Studio tool window, not a standalone SDK executable.
// No executable paths or launch arguments are accepted from the renderer.
export function openAndroidDeviceManager(): Promise<void> {
  pending ??= openStudio().finally(() => {
    pending = undefined;
  });
  return pending;
}

async function openStudio(): Promise<void> {
  try {
    let executable = androidStudioCandidates(process.platform, process.env, os.homedir()).find((candidate) =>
      fs.existsSync(candidate),
    );
    if (!executable) {
      const result = await dialog.showOpenDialog({
        title: 'Locate Android Studio — install Android Studio first if needed',
        properties: ['openFile'],
        filters: [
          {
            name: 'Android Studio',
            extensions: process.platform === 'win32' ? ['exe'] : process.platform === 'darwin' ? ['app'] : ['sh'],
          },
        ],
      });
      if (result.canceled || !result.filePaths[0]) return;
      executable = result.filePaths[0];
    }
    const confirmation = await dialog.showMessageBox({
      type: 'info',
      title: 'Android Device Manager',
      message: 'Manage emulators in Android Studio',
      detail:
        'On the welcome screen, choose More Actions → Virtual Device Manager. With a project open, choose View → Tool Windows → Device Manager, then Create Virtual Device. ARC automatically discovers newly created emulators.',
      buttons: ['Open Android Studio', 'Cancel'],
      defaultId: 0,
      cancelId: 1,
    });
    if (confirmation.response !== 0) return;
    await new Promise<void>((resolve, reject) => {
      const child = spawn(
        process.platform === 'darwin' ? '/usr/bin/open' : executable,
        process.platform === 'darwin' ? ['-a', executable] : [],
        {
          shell: false,
          windowsHide: true,
          detached: true,
          stdio: 'ignore',
        },
      );
      child.once('error', reject);
      child.once('spawn', () => {
        child.unref();
        resolve();
      });
    });
  } catch (error) {
    dialog.showErrorBox('Cannot open Android Device Manager', error instanceof Error ? error.message : String(error));
  }
}
