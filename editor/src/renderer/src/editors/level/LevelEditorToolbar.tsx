import type { AndroidTargetController } from '../../app/useAndroidTargets';
import type { AndroidTarget } from '../../../../common/androidDeviceTypes';
import { MainToolbar, type EditorTargetDevice, type MainToolbarProps } from '../../layout/MainToolbar';
import { UiButton } from '../../ui';

export type LevelEditorToolbarProps = MainToolbarProps & { androidTargets?: AndroidTargetController };

const statusLabel = (target: AndroidTarget): string => {
  if (target.ready) return 'Ready';
  const labels = {
    online: 'Connecting',
    offline: 'Offline',
    unauthorized: 'Unauthorized',
    'no-permissions': 'No permissions',
    other: 'Unavailable',
    stopped: 'Stopped',
    starting: 'Starting',
    failed: 'Failed',
  };
  return labels[target.state];
};

export function LevelEditorToolbar({ androidTargets, ...props }: LevelEditorToolbarProps) {
  if (props.targetPlatform !== 'android' || props.targetDevices || !androidTargets) return <MainToolbar {...props} />;
  const { snapshot, selected, starting, error, select, start } = androidTargets;
  const current =
    selected?.sdkRoot === snapshot?.sdkRoot
      ? snapshot?.targets.find((target) => target.id === selected?.id)
      : undefined;
  const devices: EditorTargetDevice[] = (snapshot?.targets ?? []).map((target) => ({
    id: target.id,
    label: target.name,
    menuLabel: `${[target.manufacturer, target.name].filter(Boolean).join(' - ')}${target.androidVersion ? ` (Android ${target.androidVersion})` : target.apiLevel ? ` (API ${target.apiLevel})` : ''}`,
    details: [
      target.serial ?? (target.avdName ? `AVD: ${target.avdName}` : 'Serial unavailable'),
      <span key="status">
        <span
          className={`toolbar-target-status is-${target.ready ? 'ready' : ['starting', 'stopped', 'unauthorized', 'online'].includes(target.state) ? 'pending' : 'error'}`}
        >
          {statusLabel(target)}
        </span>
        {target.kind !== 'emulator' && target.connectionType === 'network' && target.networkAddress
          ? ` (${target.networkAddress})`
          : null}
      </span>,
    ],
    platform: 'android',
    disabled: !target.ready && !target.canStart && target.state !== 'starting',
  }));
  const placeholder = !snapshot
    ? error
      ? 'Discovery unavailable'
      : 'Looking for targets…'
    : selected && !current
      ? 'Selected target unavailable'
      : !snapshot.available
        ? 'ADB unavailable'
        : 'Select Android target';
  if (!current) devices.unshift({ id: '__choose__', label: placeholder, platform: 'android', disabled: true });
  const diagnostics = [
    error,
    ...(snapshot?.diagnostics ?? []),
    ...(snapshot?.targets
      .filter((target) => target.diagnostic)
      .map((target) => `${target.name}: ${target.diagnostic}`) ?? []),
  ]
    .filter(Boolean)
    .join('\n');
  return (
    <MainToolbar
      {...props}
      targetDevices={devices}
      targetDevice={current?.id ?? '__choose__'}
      targetDeviceTitle={diagnostics || undefined}
      onTargetDeviceChange={select}
      targetDeviceMenuActions={[
        {
          value: '__android_device_manager__',
          label: 'Android Device Manager…',
          title: 'Open Android Studio to create or manage virtual devices',
          separatorBefore: true,
          onSelect: () => void window.arcAndroid?.openDeviceManager(),
        },
      ]}
      targetDeviceActions={
        <>
          {starting || current?.state === 'starting' ? (
            <span role="status">Starting emulator…</span>
          ) : current?.canStart ? (
            <UiButton onClick={() => void start()}>Start Emulator</UiButton>
          ) : null}
        </>
      }
    />
  );
}
