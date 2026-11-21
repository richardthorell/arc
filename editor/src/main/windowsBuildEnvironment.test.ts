import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, expect, it, vi } from 'vitest';
import type { EditorSettingsSnapshot } from '../common/editorWorkflowTypes';
import { resolveWindowsBuildEnvironment } from './windowsBuildWorkflow';
import { resolveVisualStudioGenerator, windowsToolchainSettingKeys } from './windowsToolchain';

vi.mock('./windowsToolchain', async (original) => ({
  ...(await original<typeof import('./windowsToolchain')>()),
  resolveVisualStudioGenerator: vi.fn(() => 'Visual Studio 17 2022'),
}));
const roots: string[] = [];
afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllEnvs();
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

it('accepts only the explicitly selected development SDK and matching CLI', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-development-sdk-'));
  roots.push(root);
  const tool = path.join(root, 'arc-project.exe');
  fs.writeFileSync(tool, 'fixture');
  const config = path.join(root, 'Release', 'lib', 'cmake', 'ARC', 'ARCConfig.cmake');
  fs.mkdirSync(path.dirname(config), { recursive: true });
  fs.writeFileSync(config, 'fixture');
  fs.writeFileSync(
    path.join(root, 'Release', 'arc-development-sdk.json'),
    JSON.stringify({
      version: 1,
      engineVersion: '0.1.0',
      architecture: 'x86_64',
      buildDirectory: root,
      projectTool: tool,
    }),
  );
  vi.stubEnv('ARC_DEVELOPMENT_SDK', root);
  vi.stubEnv('ARC_DEVELOPMENT_CONFIGURATION', 'Release');
  const settings = {
    pathValidation: {
      [windowsToolchainSettingKeys.cmakePath]: { valid: true, resolvedPath: 'cmake' },
      [windowsToolchainSettingKeys.visualStudioPath]: { valid: true, resolvedPath: 'studio' },
    },
  } as unknown as EditorSettingsSnapshot;
  expect(resolveWindowsBuildEnvironment(tool, settings)).toMatchObject({
    sdkRoot: path.join(root, 'Shipping'),
    development: { buildDirectory: root, engineVersion: '0.1.0' },
  });
  const other = path.join(root, 'other-cli.exe');
  fs.writeFileSync(other, 'other');
  expect(resolveWindowsBuildEnvironment(other, settings)).toBeUndefined();
  vi.stubEnv('ARC_DEVELOPMENT_CONFIGURATION', '../other');
  expect(resolveWindowsBuildEnvironment(tool, settings)).toBeUndefined();
});

it('uses only the selected installed SDK and configured Visual Studio instance', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-sdk-environment-'));
  roots.push(root);
  fs.mkdirSync(path.join(root, 'bin'));
  const tool = path.join(root, 'bin', 'arc-project.exe');
  fs.writeFileSync(tool, 'fixture');
  const cmake = path.join(root, 'cmake.exe');
  const studio = path.join(root, 'selected-studio');
  const settings = {
    pathValidation: {
      [windowsToolchainSettingKeys.cmakePath]: { valid: true, resolvedPath: cmake },
      [windowsToolchainSettingKeys.visualStudioPath]: { valid: true, resolvedPath: studio },
    },
  } as unknown as EditorSettingsSnapshot;
  expect(resolveWindowsBuildEnvironment(tool, settings)).toBeUndefined();
  expect(resolveVisualStudioGenerator).not.toHaveBeenCalled();
  const config = path.join(root, 'lib', 'cmake', 'ARC', 'ARCConfig.cmake');
  fs.mkdirSync(path.dirname(config), { recursive: true });
  fs.writeFileSync(config, 'fixture');
  expect(resolveWindowsBuildEnvironment(tool, settings)).toEqual({
    cmake,
    sdkRoot: fs.realpathSync(root),
    visualStudio: studio,
    generator: 'Visual Studio 17 2022',
  });
  expect(resolveVisualStudioGenerator).toHaveBeenCalledWith(studio);
  expect(resolveWindowsBuildEnvironment(tool, null)).toBeUndefined();
});
