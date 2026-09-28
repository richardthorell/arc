import { useEffect, useState } from 'react';

import type { AndroidDevice, AndroidDeviceSnapshot } from '../../../../common/androidDeviceTypes';
import { MainToolbar, type EditorTargetDevice, type MainToolbarProps } from '../../layout/MainToolbar';

export type LevelEditorToolbarProps = MainToolbarProps;

const connectionLabel = (device: AndroidDevice): string => {
  if (device.state === 'online') return device.apiLevel ? `API ${device.apiLevel}` : '';
  if (device.state === 'offline') return 'Offline';
  if (device.state === 'unauthorized') return 'Unauthorized';
  if (device.state === 'no-permissions') return 'No permissions';
  return device.stateDetail || 'Unavailable';
};

const targetDeviceFromAndroid = (device: AndroidDevice): EditorTargetDevice => {
  const name = device.model || device.serial;
  const detail = connectionLabel(device);
  return {
    id: device.serial,
    label: detail ? `${name} · ${detail}` : name,
    platform: 'android',
    disabled: device.state !== 'online',
  };
};

export function LevelEditorToolbar(props: LevelEditorToolbarProps) {
  const [androidDevices, setAndroidDevices] = useState<AndroidDeviceSnapshot | null>(null);
  const discoverAndroidDevices = props.targetPlatform === 'android' && props.targetDevices === undefined;

  useEffect(() => {
    if (!discoverAndroidDevices) {
      setAndroidDevices(null);
      return;
    }

    let disposed = false;
    let refreshing = false;
    const refresh = async () => {
      if (refreshing) return;
      refreshing = true;
      try {
        const snapshot = await window.arcAndroid?.listDevices();
        if (!disposed && snapshot) setAndroidDevices(snapshot);
      } catch {
        if (!disposed)
          setAndroidDevices({
            available: false,
            adbPath: '',
            adbVersion: '',
            devices: [],
            error: 'ADB device discovery failed',
            refreshedAt: new Date().toISOString(),
          });
      } finally {
        refreshing = false;
      }
    };

    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [discoverAndroidDevices]);

  let discoveredTargetDevices: EditorTargetDevice[] | undefined;
  if (discoverAndroidDevices) {
    if (!androidDevices) {
      discoveredTargetDevices = [
        { id: '__discovering__', label: 'Looking for devices…', platform: 'android', disabled: true },
      ];
    } else if (!androidDevices.available) {
      discoveredTargetDevices = [
        { id: '__adb-unavailable__', label: 'ADB unavailable', platform: 'android', disabled: true },
      ];
    } else {
      discoveredTargetDevices = androidDevices.devices
        .map(targetDeviceFromAndroid)
        .sort((left, right) => Number(Boolean(left.disabled)) - Number(Boolean(right.disabled)) || left.label.localeCompare(right.label));
    }
  }

  return <MainToolbar {...props} targetDevices={props.targetDevices ?? discoveredTargetDevices} />;
}
