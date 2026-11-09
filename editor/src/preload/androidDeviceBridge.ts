import { contextBridge, ipcRenderer } from 'electron';

import type { ArcAndroidBridge } from '../common/androidDeviceTypes';

const bridge: ArcAndroidBridge = {
  listDevices: () => ipcRenderer.invoke('android:list-targets'),
  listTargets: () => ipcRenderer.invoke('android:list-targets'),
  startEmulator: (target) => ipcRenderer.invoke('android:start-emulator', target),
  resolveTarget: (target) => ipcRenderer.invoke('android:resolve-target', target),
};

contextBridge.exposeInMainWorld('arcAndroid', bridge);
