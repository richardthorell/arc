import fs from 'node:fs';
import path from 'node:path';

export type AndroidSdkManagerLauncher = {
  kind: 'gui' | 'cli';
  command: string;
  args: string[];
  cwd: string;
};

const fileExists = (filePath: string): boolean => {
  try {
    return fs.statSync(filePath).isFile();
  } catch {
    return false;
  }
};

const directoryExists = (directoryPath: string): boolean => {
  try {
    return fs.statSync(directoryPath).isDirectory();
  } catch {
    return false;
  }
};

const versionParts = (value: string): number[] => value.split(/[.-]/).map((part) => Number.parseInt(part, 10) || 0);

const compareVersionsDescending = (left: string, right: string): number => {
  const a = versionParts(left);
  const b = versionParts(right);
  const count = Math.max(a.length, b.length);
  for (let index = 0; index < count; ++index) {
    const delta = (b[index] ?? 0) - (a[index] ?? 0);
    if (delta !== 0) return delta;
  }
  return right.localeCompare(left);
};

const commandLineToolRoots = (sdkRoot: string): string[] => {
  const root = path.join(sdkRoot, 'cmdline-tools');
  const result = [path.join(root, 'latest')];
  if (directoryExists(root)) {
    try {
      result.push(
        ...fs
          .readdirSync(root, { withFileTypes: true })
          .filter((entry) => entry.isDirectory() && entry.name !== 'latest')
          .map((entry) => entry.name)
          .sort(compareVersionsDescending)
          .map((entry) => path.join(root, entry)),
      );
    } catch {
      // Keep the conventional latest candidate.
    }
  }
  return result;
};

export const resolveAndroidSdkManagerLauncher = (
  sdkRoot: string,
  platform: NodeJS.Platform = process.platform,
): AndroidSdkManagerLauncher | null => {
  const resolvedRoot = path.resolve(sdkRoot);
  if (platform === 'win32') {
    const legacyGui = path.join(resolvedRoot, 'SDK Manager.exe');
    if (fileExists(legacyGui)) return { kind: 'gui', command: legacyGui, args: [], cwd: resolvedRoot };
  }

  const executableSuffix = platform === 'win32' ? '.bat' : '';
  for (const toolRoot of commandLineToolRoots(resolvedRoot)) {
    const androidCli = path.join(toolRoot, 'bin', `android${executableSuffix}`);
    if (fileExists(androidCli)) {
      return {
        kind: 'cli',
        command: androidCli,
        args: [`--sdk=${resolvedRoot}`, 'sdk', 'list'],
        cwd: resolvedRoot,
      };
    }
    const sdkManager = path.join(toolRoot, 'bin', `sdkmanager${executableSuffix}`);
    if (fileExists(sdkManager)) {
      return {
        kind: 'cli',
        command: sdkManager,
        args: [`--sdk_root=${resolvedRoot}`, '--list'],
        cwd: resolvedRoot,
      };
    }
  }

  const legacyCli = path.join(resolvedRoot, 'tools', 'bin', `sdkmanager${executableSuffix}`);
  return fileExists(legacyCli)
    ? {
        kind: 'cli',
        command: legacyCli,
        args: [`--sdk_root=${resolvedRoot}`, '--list'],
        cwd: resolvedRoot,
      }
    : null;
};
