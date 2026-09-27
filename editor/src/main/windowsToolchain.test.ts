import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { resolveWindowsToolchainValidation, windowsToolchainSettingKeys } from './windowsToolchain';

const roots: string[] = [];

const executableName = (name: string): string => (process.platform === 'win32' ? `${name}.exe` : name);

const touch = (filePath: string, contents = ''): void => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, contents, 'utf8');
};

const makeVisualStudio = (root: string): void => {
  touch(path.join(root, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat'));
};

const makeMsvcToolchain = (root: string): void => {
  touch(path.join(root, 'bin', 'Hostx64', 'x64', 'cl.exe'));
};

const makeWindowsSdk = (root: string, version = '10.0.26100.0'): void => {
  touch(path.join(root, 'Include', version, 'um', 'Windows.h'));
};

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('resolveWindowsToolchainValidation', () => {
  it('validates configured Visual Studio, MSVC, Windows SDK, CMake, and Ninja paths', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-windows-toolchain-'));
    roots.push(root);
    const visualStudioPath = path.join(root, 'VisualStudio');
    const msvcToolchainPath = path.join(root, 'msvc', '14.44.35207');
    const sdkPath = path.join(root, 'WindowsSdk');
    const cmakePath = path.join(root, 'bin', executableName('cmake'));
    const ninjaPath = path.join(root, 'bin', executableName('ninja'));
    makeVisualStudio(visualStudioPath);
    makeMsvcToolchain(msvcToolchainPath);
    makeWindowsSdk(sdkPath);
    touch(cmakePath);
    touch(ninjaPath);

    const validation = resolveWindowsToolchainValidation(
      { visualStudioPath, msvcToolchainPath, sdkPath, cmakePath, ninjaPath },
      { PATH: '' },
    );

    expect(validation[windowsToolchainSettingKeys.visualStudioPath]).toMatchObject({
      valid: true,
      resolvedPath: visualStudioPath,
      source: 'configured',
    });
    expect(validation[windowsToolchainSettingKeys.msvcToolchainPath]).toMatchObject({
      valid: true,
      resolvedPath: msvcToolchainPath,
      source: 'configured',
    });
    expect(validation[windowsToolchainSettingKeys.sdkPath]).toMatchObject({
      valid: true,
      resolvedPath: sdkPath,
      source: 'configured',
    });
    expect(validation[windowsToolchainSettingKeys.cmakePath]).toMatchObject({
      valid: true,
      resolvedPath: cmakePath,
      source: 'configured',
    });
    expect(validation[windowsToolchainSettingKeys.ninjaPath]).toMatchObject({
      valid: true,
      resolvedPath: ninjaPath,
      source: 'configured',
    });
  });

  it('auto-detects environment, derived MSVC, and PATH toolchain locations', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-windows-toolchain-'));
    roots.push(root);
    const visualStudioPath = path.join(root, 'VisualStudio');
    const msvcToolchainPath = path.join(visualStudioPath, 'VC', 'Tools', 'MSVC', '14.44.35207');
    const sdkPath = path.join(root, 'WindowsSdk');
    const binPath = path.join(root, 'bin');
    const cmakePath = path.join(binPath, executableName('cmake'));
    const ninjaPath = path.join(binPath, executableName('ninja'));
    makeVisualStudio(visualStudioPath);
    makeMsvcToolchain(msvcToolchainPath);
    makeWindowsSdk(sdkPath, '10.0.22621.0');
    touch(cmakePath);
    touch(ninjaPath);

    const validation = resolveWindowsToolchainValidation(
      {},
      { VSINSTALLDIR: visualStudioPath, WindowsSdkDir: sdkPath, PATH: binPath },
    );

    expect(validation[windowsToolchainSettingKeys.visualStudioPath]).toMatchObject({
      valid: true,
      resolvedPath: visualStudioPath,
      source: 'environment',
    });
    expect(validation[windowsToolchainSettingKeys.msvcToolchainPath]).toMatchObject({
      valid: true,
      resolvedPath: msvcToolchainPath,
      source: 'derived',
    });
    expect(validation[windowsToolchainSettingKeys.sdkPath].message).toContain('10.0.22621.0');
    expect(validation[windowsToolchainSettingKeys.cmakePath]).toMatchObject({
      valid: true,
      resolvedPath: cmakePath,
      source: 'environment',
    });
    expect(validation[windowsToolchainSettingKeys.ninjaPath]).toMatchObject({
      valid: true,
      resolvedPath: ninjaPath,
      source: 'environment',
    });
  });

  it('reports invalid configured paths instead of silently falling back to detected tools', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-windows-toolchain-'));
    roots.push(root);
    const missing = path.join(root, 'missing');
    const binPath = path.join(root, 'bin');
    touch(path.join(binPath, executableName('cmake')));
    touch(path.join(binPath, executableName('ninja')));

    const validation = resolveWindowsToolchainValidation(
      {
        visualStudioPath: missing,
        msvcToolchainPath: missing,
        sdkPath: missing,
        cmakePath: missing,
        ninjaPath: missing,
      },
      { PATH: binPath },
    );

    for (const key of Object.values(windowsToolchainSettingKeys)) {
      expect(validation[key]).toMatchObject({ valid: false, source: 'configured' });
    }
  });
});
