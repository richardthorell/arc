import { contextBridge, ipcRenderer } from 'electron';
import fs from 'node:fs';
import path from 'node:path';

import type { AndroidDeviceSnapshot, ArcAndroidBridge } from '../common/androidDeviceTypes';
import type { EditorSettingsSnapshot } from '../common/editorWorkflowTypes';
import { AdbClient } from './adbClient';

const androidSdkSettingKey = 'platform.android.sdkPath';

const adbExecutableName = (): string => (process.platform === 'win32' ? 'adb.exe' : 'adb');

export const resolveAndroidAdbPath = (snapshot: EditorSettingsSnapshot | null | undefined): string => {
  const validation = snapshot?.pathValidation?.[androidSdkSettingKey];
  const sdkRoot =
    validation?.valid && validation.resolvedPath
      ? validation.resolvedPath
      : typeof snapshot?.values?.[androidSdkSettingKey] === 'string'
        ? String(snapshot.values[androidSdkSettingKey]).trim()
        : '';
  if (!sdkRoot) return '';
  return path.join(sdkRoot, 'platform-tools', adbExecutableName());
};

let cachedClient: AdbClient | null = null;

const unavailableSnapshot = (adbPath: string, error: string): AndroidDeviceSnapshot => ({
  available: false,
  adbPath,
  adbVersion: '',
  devices: [],
  error,
  refreshedAt: new Date().toISOString(),
});

const listDevices = async (): Promise<AndroidDeviceSnapshot> => {
  const settings = (await ipcRenderer.invoke('settings:snapshot')) as EditorSettingsSnapshot | null;
  const adbPath = resolveAndroidAdbPath(settings);
  if (!adbPath) return unavailableSnapshot('', 'Android SDK / ADB is not configured');
  if (!fs.existsSync(adbPath)) return unavailableSnapshot(adbPath, 'ADB executable was not found in the configured Android SDK');

  if (!cachedClient || cachedClient.executable !== adbPath) cachedClient = new AdbClient(adbPath);

  try {
    const [adbVersion, devices] = await Promise.all([cachedClient.version(), cachedClient.listDevices()]);
    return {
      available: true,
      adbPath,
      adbVersion,
      devices,
      error: '',
      refreshedAt: new Date().toISOString(),
    };
  } catch (error) {
    return unavailableSnapshot(adbPath, error instanceof Error ? error.message : String(error));
  }
};

const bridge: ArcAndroidBridge = {
  listDevices,
};

contextBridge.exposeInMainWorld('arcAndroid', bridge);
