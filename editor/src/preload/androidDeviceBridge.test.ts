import path from 'node:path';

import { describe, expect, it, vi } from 'vitest';

vi.mock('electron', () => ({
  contextBridge: { exposeInMainWorld: vi.fn() },
  ipcRenderer: { invoke: vi.fn() },
}));

import type { EditorSettingsSnapshot } from '../common/editorWorkflowTypes';
import { resolveAndroidAdbPath } from './androidDeviceBridge';

describe('Android device bridge', () => {
  it('uses the validated Android SDK path as the ADB authority', () => {
    const snapshot = {
      values: { 'platform.android.sdkPath': '/configured-but-invalid' },
      pathValidation: {
        'platform.android.sdkPath': {
          valid: true,
          resolvedPath: '/detected/android-sdk',
          message: 'Validated · Android SDK / ADB',
          source: 'environment',
        },
      },
    } as unknown as EditorSettingsSnapshot;

    expect(resolveAndroidAdbPath(snapshot)).toBe(
      path.join('/detected/android-sdk', 'platform-tools', process.platform === 'win32' ? 'adb.exe' : 'adb'),
    );
  });

  it('falls back to the effective settings value when path validation is unavailable', () => {
    const snapshot = {
      values: { 'platform.android.sdkPath': '/android-sdk' },
    } as unknown as EditorSettingsSnapshot;

    expect(resolveAndroidAdbPath(snapshot)).toBe(
      path.join('/android-sdk', 'platform-tools', process.platform === 'win32' ? 'adb.exe' : 'adb'),
    );
  });
});
