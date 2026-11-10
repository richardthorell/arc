import path from 'node:path';

import { describe, expect, it, vi } from 'vitest';

vi.mock('electron', () => ({
  contextBridge: { exposeInMainWorld: vi.fn() },
  ipcRenderer: { invoke: vi.fn() },
}));

import type { EditorSettingsSnapshot } from '../common/editorWorkflowTypes';
import { resolveAndroidAdbPath } from '../main/androidTargetService';
import { contextBridge, ipcRenderer } from 'electron';
import './androidDeviceBridge';
import type { ArcAndroidBridge } from '../common/androidDeviceTypes';

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
      path.resolve('/detected/android-sdk', 'platform-tools', process.platform === 'win32' ? 'adb.exe' : 'adb'),
    );
  });

  it('falls back to the effective settings value when path validation is unavailable', () => {
    const snapshot = {
      values: { 'platform.android.sdkPath': '/android-sdk' },
    } as unknown as EditorSettingsSnapshot;

    expect(resolveAndroidAdbPath(snapshot)).toBe(
      path.resolve('/android-sdk', 'platform-tools', process.platform === 'win32' ? 'adb.exe' : 'adb'),
    );
  });
  it('exposes only narrow main-process operations, preserving listDevices', async () => {
    const bridge = vi.mocked(contextBridge.exposeInMainWorld).mock.calls[0][1] as ArcAndroidBridge;
    const target = { sdkRoot: '/sdk', id: 'avd:Pixel' };
    await bridge.listDevices();
    await bridge.listTargets();
    await bridge.startEmulator(target);
    await bridge.resolveTarget(target);
    await bridge.openDeviceManager();
    expect(ipcRenderer.invoke).toHaveBeenCalledWith('android:open-device-manager');
    expect(ipcRenderer.invoke).toHaveBeenCalledWith('android:list-targets');
    expect(ipcRenderer.invoke).toHaveBeenCalledWith('android:start-emulator', target);
    expect(ipcRenderer.invoke).toHaveBeenCalledWith('android:resolve-target', target);
    expect(Object.keys(bridge).sort()).toEqual([
      'listDevices',
      'listTargets',
      'openDeviceManager',
      'resolveTarget',
      'startEmulator',
    ]);
  });
});
