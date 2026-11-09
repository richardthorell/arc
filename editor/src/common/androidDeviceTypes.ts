export type AndroidDeviceConnectionState = 'online' | 'offline' | 'unauthorized' | 'no-permissions' | 'other';

export type AndroidDevice = {
  serial: string;
  state: AndroidDeviceConnectionState;
  stateDetail: string;
  model: string;
  product: string;
  device: string;
  transportId: string;
  apiLevel?: number;
  emulator: boolean;
};

export type AndroidDeviceSnapshot = {
  available: boolean;
  adbPath: string;
  adbVersion: string;
  devices: AndroidDevice[];
  error: string;
  refreshedAt: string;
};

export type ArcAndroidBridge = {
  listDevices(): Promise<AndroidDeviceSnapshot>;
  listTargets(): Promise<AndroidTargetSnapshot>;
  startEmulator(target: AndroidTargetReference): Promise<void>;
  resolveTarget(target: AndroidTargetReference): Promise<{ serial: string; sdkRoot: string }>;
};

export type AndroidTargetReference = { sdkRoot: string; id: string };
export type AndroidTarget = {
  id: string;
  kind: 'device' | 'emulator';
  name: string;
  serial?: string;
  avdName?: string;
  apiLevel?: number;
  state: AndroidDeviceConnectionState | 'stopped' | 'starting' | 'failed';
  ready: boolean;
  canStart: boolean;
  diagnostic?: string;
};

export type AndroidTargetSnapshot = AndroidDeviceSnapshot & {
  sdkRoot: string;
  targets: AndroidTarget[];
  emulatorAvailable: boolean;
  diagnostics: string[];
};

declare global {
  interface Window {
    arcAndroid?: ArcAndroidBridge;
  }
}
