import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { resolveAndroidSdkManagerLauncher } from './androidSdkManager';

const roots: string[] = [];

const touch = (filePath: string): void => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, '', 'utf8');
};

const makeRoot = (): string => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-android-sdk-manager-'));
  roots.push(root);
  return root;
};

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('resolveAndroidSdkManagerLauncher', () => {
  it('prefers the legacy Windows SDK Manager GUI when present', () => {
    const root = makeRoot();
    const executable = path.join(root, 'SDK Manager.exe');
    touch(executable);
    touch(path.join(root, 'cmdline-tools', 'latest', 'bin', 'sdkmanager.bat'));

    expect(resolveAndroidSdkManagerLauncher(root, 'win32')).toEqual({
      kind: 'gui',
      command: executable,
      args: [],
      cwd: root,
    });
  });

  it('uses the current Android CLI when command-line tools provide it', () => {
    const root = makeRoot();
    const executable = path.join(root, 'cmdline-tools', 'latest', 'bin', 'android.bat');
    touch(executable);

    expect(resolveAndroidSdkManagerLauncher(root, 'win32')).toEqual({
      kind: 'cli',
      command: executable,
      args: [`--sdk=${root}`, 'sdk', 'list'],
      cwd: root,
    });
  });

  it('falls back to the sdkmanager command-line tool', () => {
    const root = makeRoot();
    const executable = path.join(root, 'cmdline-tools', '22.0', 'bin', 'sdkmanager.bat');
    touch(executable);

    expect(resolveAndroidSdkManagerLauncher(root, 'win32')).toEqual({
      kind: 'cli',
      command: executable,
      args: [`--sdk_root=${root}`, '--list'],
      cwd: root,
    });
  });

  it('returns null when the validated SDK has no manager tooling', () => {
    expect(resolveAndroidSdkManagerLauncher(makeRoot(), 'win32')).toBeNull();
  });
});
