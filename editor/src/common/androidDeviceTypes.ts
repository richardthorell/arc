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
};

declare global {
  interface Window {
    arcAndroid?: ArcAndroidBridge;
  }
}
