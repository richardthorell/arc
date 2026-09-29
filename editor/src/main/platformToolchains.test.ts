import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import {
  appleToolchainSettingKeys,
  buildToolSettingKeys,
  linuxToolchainSettingKeys,
  resolveAppleToolchainValidation,
  resolveBuildToolValidation,
  resolveLinuxToolchainValidation,
  resolveWebToolchainValidation,
  webToolchainSettingKeys,
} from './platformToolchains';

const roots: string[] = [];

const touch = (filePath: string): void => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, '', 'utf8');
};

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('platform toolchains', () => {
  it('validates common CMake and Ninja overrides independently of the target platform', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-build-tools-'));
    roots.push(root);
    const cmake = path.join(root, 'cmake');
    const ninja = path.join(root, 'ninja');
    touch(cmake);
    touch(ninja);

    const validation = resolveBuildToolValidation({ cmakePath: cmake, ninjaPath: ninja }, { PATH: '' }, 'linux');

    expect(validation[buildToolSettingKeys.cmakePath]).toMatchObject({
      valid: true,
      resolvedPath: cmake,
      source: 'configured',
    });
    expect(validation[buildToolSettingKeys.ninjaPath]).toMatchObject({
      valid: true,
      resolvedPath: ninja,
      source: 'configured',
    });
  });

  it('discovers a Linux compiler and sysroot through WSL on Windows', () => {
    const runner = (_executable: string, arguments_: string[]): string => {
      const command = arguments_.join(' ');
      if (command === '-l -q') return 'Ubuntu-24.04\n';
      if (command.includes('command -v clang++')) return '/usr/bin/clang++\n';
      if (command.includes("test -x '/usr/bin/clang++'")) return 'ok\n';
      if (command.includes("test -d '/'")) return 'ok\n';
      throw new Error(`Unexpected command: ${command}`);
    };

    const validation = resolveLinuxToolchainValidation(
      { buildEnvironment: 'wsl' },
      { SystemRoot: 'C:\\Windows', PATH: '' },
      'win32',
      runner,
    );

    expect(validation[linuxToolchainSettingKeys.compilerPath]).toMatchObject({
      valid: true,
      resolvedPath: '/usr/bin/clang++',
      source: 'derived',
    });
    expect(validation[linuxToolchainSettingKeys.compilerPath].message).toContain('Ubuntu-24.04');
    expect(validation[linuxToolchainSettingKeys.sysrootPath]).toMatchObject({ valid: true, resolvedPath: '/' });
  });

  it('explains that Apple SDK discovery requires a macOS host', () => {
    const validation = resolveAppleToolchainValidation({}, {}, 'win32');

    expect(validation[appleToolchainSettingKeys.xcodePath]).toMatchObject({ valid: false, source: 'unresolved' });
    expect(validation[appleToolchainSettingKeys.xcodePath].message).toContain('macOS');
    expect(validation[appleToolchainSettingKeys.iosDeviceSdkPath].message).toContain('macOS');
  });

  it('validates an Emscripten SDK root', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-emsdk-'));
    roots.push(root);
    touch(path.join(root, 'upstream', 'emscripten', 'emcc'));

    const validation = resolveWebToolchainValidation({ emsdkPath: root }, {});

    expect(validation[webToolchainSettingKeys.emsdkPath]).toMatchObject({
      valid: true,
      resolvedPath: root,
      source: 'configured',
    });
  });
});
