import type { AndroidTargetController } from '../../app/useAndroidTargets';
import { MainToolbar, type EditorTargetDevice, type MainToolbarProps } from '../../layout/MainToolbar';
import { UiButton } from '../../ui';

export type LevelEditorToolbarProps = MainToolbarProps & { androidTargets?: AndroidTargetController };

export function LevelEditorToolbar({ androidTargets, ...props }: LevelEditorToolbarProps) {
  if (props.targetPlatform !== 'android' || props.targetDevices || !androidTargets) return <MainToolbar {...props} />;
  const { snapshot, selected, starting, error, select, start, refresh } = androidTargets;
  const current =
    selected?.sdkRoot === snapshot?.sdkRoot
      ? snapshot?.targets.find((target) => target.id === selected?.id)
      : undefined;
  const devices: EditorTargetDevice[] = (snapshot?.targets ?? []).map((target) => ({
    id: target.id,
    label: [
      target.name,
      target.kind === 'device' ? 'Device' : 'Emulator',
      ...(target.apiLevel ? [`API ${target.apiLevel}`] : []),
      target.ready ? 'Ready' : target.state,
      ...(target.serial ? [target.serial] : []),
    ].join(' · '),
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
      onTargetDeviceChange={select}
      targetDeviceActions={
        <>
          {starting || current?.state === 'starting' ? (
            <span role="status">Starting emulator…</span>
          ) : current?.canStart ? (
            <UiButton onClick={() => void start()}>Start Emulator</UiButton>
          ) : null}
          <UiButton aria-label="Refresh Android targets" onClick={refresh}>
            Refresh
          </UiButton>
          {diagnostics && (
            <span className="toolbar-android-diagnostic" role="status" title={diagnostics} aria-label={diagnostics}>
              Android: {diagnostics}
            </span>
          )}
        </>
      }
    />
  );
}
