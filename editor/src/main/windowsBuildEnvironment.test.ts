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
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
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
