import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { androidToolchainSettingKeys, resolveAndroidToolchainValidation } from './androidToolchain';

const roots: string[] = [];

const executableName = (name: string): string => (process.platform === 'win32' ? `${name}.exe` : name);

const touch = (filePath: string, contents = ''): void => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, contents, 'utf8');
};

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('resolveAndroidToolchainValidation', () => {
  it('validates configured Java, SDK, and NDK roots without requiring fixed versions', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-android-toolchain-'));
    roots.push(root);
    const javaHome = path.join(root, 'jdk');
    const sdkPath = path.join(root, 'sdk');
    const ndkPath = path.join(sdkPath, 'ndk', '27.2.12479018');

    touch(path.join(javaHome, 'bin', executableName('java')));
    touch(path.join(javaHome, 'bin', executableName('javac')));
    touch(path.join(javaHome, 'release'), 'JAVA_VERSION="21.0.5"\n');
    touch(path.join(sdkPath, 'platform-tools', executableName('adb')));
    touch(path.join(ndkPath, 'source.properties'), 'Pkg.Revision = 27.2.12479018\n');
    touch(path.join(ndkPath, 'build', 'cmake', 'android.toolchain.cmake'));

    const validation = resolveAndroidToolchainValidation(
      { javaHome, sdkPath, ndkPath },
      { PATH: '' },
    );

    expect(validation[androidToolchainSettingKeys.javaHome]).toMatchObject({
      valid: true,
      resolvedPath: javaHome,
      source: 'configured',
    });
    expect(validation[androidToolchainSettingKeys.javaHome].message).toContain('Java 21.0.5');
    expect(validation[androidToolchainSettingKeys.sdkPath]).toMatchObject({
      valid: true,
      resolvedPath: sdkPath,
      source: 'configured',
    });
    expect(validation[androidToolchainSettingKeys.ndkPath].message).toContain('27.2.12479018');
  });

  it('auto-detects environment paths and the newest installed NDK under the SDK', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-android-toolchain-'));
    roots.push(root);
    const javaHome = path.join(root, 'jdk');
    const sdkPath = path.join(root, 'sdk');
    const olderNdk = path.join(sdkPath, 'ndk', '26.3.11579264');
    const newerNdk = path.join(sdkPath, 'ndk', '27.1.12297006');

    touch(path.join(javaHome, 'bin', executableName('java')));
    touch(path.join(javaHome, 'bin', executableName('javac')));
    touch(path.join(sdkPath, 'platform-tools', executableName('adb')));
    for (const [ndkPath, version] of [
      [olderNdk, '26.3.11579264'],
      [newerNdk, '27.1.12297006'],
    ] as const) {
      touch(path.join(ndkPath, 'source.properties'), `Pkg.Revision = ${version}\n`);
      touch(path.join(ndkPath, 'build', 'cmake', 'android.toolchain.cmake'));
    }

    const validation = resolveAndroidToolchainValidation(
      {},
      { JAVA_HOME: javaHome, ANDROID_SDK_ROOT: sdkPath, PATH: '' },
    );

    expect(validation[androidToolchainSettingKeys.javaHome].source).toBe('environment');
    expect(validation[androidToolchainSettingKeys.sdkPath].source).toBe('environment');
    expect(validation[androidToolchainSettingKeys.ndkPath]).toMatchObject({
      valid: true,
      resolvedPath: newerNdk,
      source: 'derived',
    });
  });

  it('reports invalid configured paths instead of treating non-empty strings as valid', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-android-toolchain-'));
    roots.push(root);
    const missing = path.join(root, 'missing');

    const validation = resolveAndroidToolchainValidation(
      { javaHome: missing, sdkPath: missing, ndkPath: missing },
      { PATH: '' },
    );

    expect(validation[androidToolchainSettingKeys.javaHome]).toMatchObject({ valid: false, source: 'configured' });
    expect(validation[androidToolchainSettingKeys.sdkPath]).toMatchObject({ valid: false, source: 'configured' });
    expect(validation[androidToolchainSettingKeys.ndkPath]).toMatchObject({ valid: false, source: 'configured' });
  });
});
