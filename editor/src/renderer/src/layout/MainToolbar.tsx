import { useEffect, useRef, useState } from 'react';
import {
  Box,
  Check,
  ChevronDown,
  CircleDot,
  Crosshair,
  Globe,
  Grid3X3,
  Hammer,
  Monitor,
  Mountain,
  MousePointer2,
  Move,
  Pause,
  Play,
  RefreshCw,
  Rotate3D,
  Scaling,
  Settings2,
  Smartphone,
  Square,
  StepForward,
  Trash2,
} from 'lucide-react';

import type { ArcProjectDescriptor } from '../../../common/projectTypes';
import type { CommandId, EditorRuntimeState } from '../app/workbenchTypes';
import { requestSettingsDialog } from '../settings/settingsDialogRoute';
import {
  UiButton,
  UiDropdown,
  UiIconButton,
  UiSplitButton,
  type UiDropdownOption,
  type UiSplitButtonOption,
} from '../ui';
import { PlatformBrandIcon } from './PlatformBrandIcon';

import './MainToolbar.css';

export type EditorTargetPlatform =
  'windows' | 'linux' | 'macos' | 'ios' | 'android' | 'xbox' | 'playstation' | 'switch';
export type ToolbarBuildAction = 'build' | 'rebuild' | 'configure' | 'clean';
export type ToolbarCoordinateSpace = 'world' | 'local';
export type ToolbarTransformOrigin = 'pivot' | 'center';

export type EditorTargetDevice = {
  id: string;
  label: string;
  platform: EditorTargetPlatform;
  disabled?: boolean;
};

type PlatformMenuValue = EditorTargetPlatform | '__platform-settings__';

const platformOptions: ReadonlyArray<UiDropdownOption<EditorTargetPlatform>> = [
  { value: 'windows', label: 'Windows', icon: <PlatformBrandIcon platform="windows" /> },
  { value: 'linux', label: 'Linux', icon: <PlatformBrandIcon platform="linux" /> },
  { value: 'macos', label: 'macOS', icon: <PlatformBrandIcon platform="macos" /> },
  { value: 'ios', label: 'iOS', icon: <PlatformBrandIcon platform="ios" /> },
  { value: 'android', label: 'Android', icon: <PlatformBrandIcon platform="android" /> },
  { value: 'xbox', label: 'Xbox', icon: <PlatformBrandIcon platform="xbox" /> },
  { value: 'playstation', label: 'PlayStation', icon: <PlatformBrandIcon platform="playstation" /> },
  { value: 'switch', label: 'Nintendo Switch', icon: <PlatformBrandIcon platform="switch" /> },
];

const editorTargetPlatforms = new Set<EditorTargetPlatform>(platformOptions.map((option) => option.value));

const normalizeTargetPlatform = (value: string): EditorTargetPlatform | null => {
  const normalized = value.trim().toLocaleLowerCase();
  if (editorTargetPlatforms.has(normalized as EditorTargetPlatform)) return normalized as EditorTargetPlatform;
  if (normalized === 'darwin' || normalized === 'mac') return 'macos';
  if (normalized === 'ps5' || normalized === 'ps4') return 'playstation';
  if (normalized === 'nintendo') return 'switch';
  return null;
};

const platformFromTargetId = (value: string): EditorTargetPlatform | null => {
  const normalized = value.trim().toLocaleLowerCase();
  for (const platform of editorTargetPlatforms) {
    if (normalized === platform || normalized.startsWith(`${platform}-`)) return platform;
  }
  if (normalized.startsWith('darwin-') || normalized.startsWith('mac-')) return 'macos';
  if (normalized.startsWith('ps5-') || normalized.startsWith('ps4-')) return 'playstation';
  if (normalized.startsWith('nintendo-')) return 'switch';
  return null;
};

export const detectHostTargetPlatform = (): EditorTargetPlatform => {
  const hostPlatform = navigator.platform.toLocaleLowerCase();
  if (hostPlatform.includes('mac')) return 'macos';
  if (hostPlatform.includes('linux')) return 'linux';
  return 'windows';
};

export const configuredTargetPlatformsForProject = (
  descriptor: Pick<ArcProjectDescriptor, 'targetPlatforms' | 'cookProfiles'> | null | undefined,
  hostPlatform: EditorTargetPlatform,
): EditorTargetPlatform[] => {
  if (!descriptor) return [hostPlatform];

  const enabledTargets = descriptor.targetPlatforms.filter((target) => target.enabled && target.id.trim());
  if (!enabledTargets.length) return [hostPlatform];

  const enabledIds = new Set(enabledTargets.map((target) => target.id));
  const configured: EditorTargetPlatform[] = [];
  const append = (platform: EditorTargetPlatform | null) => {
    if (platform && !configured.includes(platform)) configured.push(platform);
  };

  for (const profile of descriptor.cookProfiles) {
    if (enabledIds.has(profile.id)) append(normalizeTargetPlatform(profile.platform));
  }
  for (const target of enabledTargets) append(platformFromTargetId(target.id));

  return configured.length ? configured : [hostPlatform];
};

const transformOriginOptions: ReadonlyArray<UiDropdownOption<ToolbarTransformOrigin>> = [
  { value: 'pivot', label: 'Pivot', icon: <Crosshair size={12} /> },
  { value: 'center', label: 'Center', icon: <CircleDot size={12} /> },
];

const coordinateSpaceOptions: ReadonlyArray<UiDropdownOption<ToolbarCoordinateSpace>> = [
  { value: 'world', label: 'World', icon: <Globe size={12} /> },
  { value: 'local', label: 'Local', icon: <Box size={12} /> },
];

const translationSnapOptions: ReadonlyArray<UiDropdownOption<string>> = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 5, 10].map(
  (value) => ({ value: String(value), label: String(value) }),
);

const rotationSnapOptions: ReadonlyArray<UiDropdownOption<string>> = [1, 5, 10, 15, 30, 45, 90].map((value) => ({
  value: String(value),
  label: `${value}°`,
}));

const scaleSnapOptions: ReadonlyArray<UiDropdownOption<string>> = [0.01, 0.05, 0.1, 0.25, 0.5, 1].map((value) => ({
  value: String(value),
  label: `${Math.round(value * 100)}%`,
}));

const timeScaleOptions: ReadonlyArray<UiDropdownOption<string>> = [0.25, 0.5, 1, 2, 4].map((value) => ({
  value: String(value),
  label: `${value}×`,
}));

type ToolbarPlaybackOptionsProps = {
  timeScale: number;
  onTimeScaleChange?: (value: number) => void;
};

function ToolbarPlaybackOptions({ timeScale, onTimeScaleChange }: ToolbarPlaybackOptionsProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [open]);

  return (
    <span className="toolbar-playback-options" ref={rootRef}>
      <UiButton
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label="Playback options"
        className="toolbar-playback-options-trigger"
        onClick={() => setOpen((current) => !current)}
        type="button"
        variant="toolbar"
      >
        <ChevronDown aria-hidden="true" size={12} />
      </UiButton>
      {open && (
        <div className="toolbar-playback-options-popup" role="menu" aria-label="Playback options menu">
          <div className="toolbar-playback-options-row">
            <span>Time scale</span>
            <UiDropdown
              ariaLabel="Simulation time scale"
              className="toolbar-playback-time-scale"
              onValueChange={(value) => onTimeScaleChange?.(Number(value))}
              options={timeScaleOptions}
              value={String(timeScale)}
            />
          </div>
        </div>
      )}
    </span>
  );
}

const buildOptions: ReadonlyArray<UiSplitButtonOption<ToolbarBuildAction>> = [
  { value: 'build', label: 'Build', icon: <Hammer size={14} /> },
  { value: 'rebuild', label: 'Rebuild', icon: <RefreshCw size={14} /> },
  { value: 'configure', label: 'Configure', icon: <Settings2 size={14} /> },
  { value: 'clean', label: 'Clean', icon: <Trash2 size={14} /> },
];

type ToolbarSnapMenuProps = {
  snapping: boolean;
  translationSnap: number;
  rotationSnap: number;
  scaleSnap: number;
  onToggleSnapping?: () => void;
  onTranslationSnapChange?: (value: number) => void;
  onRotationSnapChange?: (value: number) => void;
  onScaleSnapChange?: (value: number) => void;
  disabled?: boolean;
};

function ToolbarSnapMenu({
  snapping,
  translationSnap,
  rotationSnap,
  scaleSnap,
  onToggleSnapping,
  onTranslationSnapChange,
  onRotationSnapChange,
  onScaleSnapChange,
  disabled = false,
}: ToolbarSnapMenuProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [open]);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  return (
    <span className="toolbar-snap-menu" ref={rootRef}>
      <UiButton
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label="Snap settings"
        className={`toolbar-snap-trigger${snapping ? ' is-active' : ''}`}
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
        type="button"
        variant="toolbar"
      >
        <Grid3X3 size={14} />
        <span>Snap</span>
        <ChevronDown aria-hidden="true" size={12} />
      </UiButton>
      {open && (
        <div className="toolbar-snap-popup" role="menu" aria-label="Snap settings menu">
          <UiButton
            aria-pressed={snapping}
            className="toolbar-snap-enable"
            onClick={onToggleSnapping}
            type="button"
            variant="ghost"
          >
            <span>Enable snapping</span>
            <span className="toolbar-snap-check" aria-hidden="true">
              {snapping ? <Check size={13} /> : null}
            </span>
          </UiButton>
          <div className="toolbar-snap-popup-separator" />
          <div className="toolbar-snap-row">
            <span>Move</span>
            <UiDropdown
              ariaLabel="Translation snap"
              className="toolbar-snap-value-dropdown"
              onValueChange={(value) => onTranslationSnapChange?.(Number(value))}
              options={translationSnapOptions}
              value={String(translationSnap)}
            />
          </div>
          <div className="toolbar-snap-row">
            <span>Rotate</span>
            <UiDropdown
              ariaLabel="Rotation snap"
              className="toolbar-snap-value-dropdown"
              onValueChange={(value) => onRotationSnapChange?.(Number(value))}
              options={rotationSnapOptions}
              value={String(rotationSnap)}
            />
          </div>
          <div className="toolbar-snap-row">
            <span>Scale</span>
            <UiDropdown
              ariaLabel="Scale snap"
              className="toolbar-snap-value-dropdown"
              onValueChange={(value) => onScaleSnapChange?.(Number(value))}
              options={scaleSnapOptions}
              value={String(scaleSnap)}
            />
          </div>
        </div>
      )}
    </span>
  );
}

export type MainToolbarProps = {
  onCommand: (command: CommandId) => void;
  activeTool?: 'select' | 'translate' | 'rotate' | 'scale' | 'terrain';
  terrainEnabled?: boolean;
  coordinateSpace?: ToolbarCoordinateSpace;
  snapping?: boolean;
  translationSnap?: number;
  rotationSnap?: number;
  scaleSnap?: number;
  onCoordinateSpaceChange?: (space: ToolbarCoordinateSpace) => void;
  onToggleSnapping?: () => void;
  onTranslationSnapChange?: (value: number) => void;
  onRotationSnapChange?: (value: number) => void;
  onScaleSnapChange?: (value: number) => void;
  runtimeState?: EditorRuntimeState;
  runtimeError?: string;
  authoringDisabled?: boolean;
  timeScale?: number;
  onTimeScaleChange?: (value: number) => void;
  targetPlatform?: EditorTargetPlatform;
  configuredTargetPlatforms?: ReadonlyArray<EditorTargetPlatform>;
  onTargetPlatformChange?: (platform: EditorTargetPlatform) => void;
  targetDevices?: ReadonlyArray<EditorTargetDevice>;
  targetDevice?: string;
  onTargetDeviceChange?: (deviceId: string) => void;
  onBuildAction?: (action: ToolbarBuildAction) => void;
};

export function MainToolbar({
  onCommand,
  activeTool = 'translate',
  coordinateSpace = 'world',
  snapping = false,
  translationSnap = 0.25,
  rotationSnap = 15,
  scaleSnap = 0.1,
  onCoordinateSpaceChange,
  onToggleSnapping,
  onTranslationSnapChange,
  onRotationSnapChange,
  onScaleSnapChange,
  terrainEnabled = false,
  runtimeState = 'stopped',
  runtimeError = '',
  authoringDisabled = false,
  timeScale = 1,
  onTimeScaleChange,
  targetPlatform,
  configuredTargetPlatforms,
  onTargetPlatformChange,
  targetDevices,
  targetDevice,
  onTargetDeviceChange,
  onBuildAction,
}: MainToolbarProps) {
  const [transformOrigin, setTransformOrigin] = useState<ToolbarTransformOrigin>('pivot');
  const hostPlatform = detectHostTargetPlatform();
  const [projectTargetPlatforms, setProjectTargetPlatforms] = useState<EditorTargetPlatform[]>(() => [hostPlatform]);
  const [localTargetDevice, setLocalTargetDevice] = useState('local');
  const playLabel = runtimeState === 'paused' ? 'Resume' : 'Play';
  const runtimeLabel =
    runtimeState === 'stopped'
      ? 'Authoring World'
      : `Play World: ${runtimeState[0].toUpperCase()}${runtimeState.slice(1)}`;

  useEffect(() => {
    if (configuredTargetPlatforms) {
      const configured = Array.from(new Set(configuredTargetPlatforms));
      setProjectTargetPlatforms(configured.length ? configured : [hostPlatform]);
      return;
    }

    let disposed = false;
    void window.arc?.projects
      ?.snapshot()
      .then((snapshot) => {
        if (disposed) return;
        setProjectTargetPlatforms(configuredTargetPlatformsForProject(snapshot?.activeProject?.descriptor, hostPlatform));
      })
      .catch(() => {
        if (!disposed) setProjectTargetPlatforms([hostPlatform]);
      });
    return () => {
      disposed = true;
    };
  }, [configuredTargetPlatforms, hostPlatform]);

  const requestedTargetPlatform = targetPlatform ?? hostPlatform;
  const effectiveTargetPlatform = projectTargetPlatforms.includes(requestedTargetPlatform)
    ? requestedTargetPlatform
    : (projectTargetPlatforms[0] ?? hostPlatform);

  useEffect(() => {
    if (targetPlatform && targetPlatform !== effectiveTargetPlatform) onTargetPlatformChange?.(effectiveTargetPlatform);
  }, [effectiveTargetPlatform, onTargetPlatformChange, targetPlatform]);

  const availableTargetDevices: EditorTargetDevice[] = targetDevices
    ? targetDevices.filter((device) => device.platform === effectiveTargetPlatform)
    : effectiveTargetPlatform === hostPlatform
      ? [{ id: 'local', label: 'This Computer', platform: hostPlatform }]
      : [];
  const requestedTargetDevice = targetDevice ?? localTargetDevice;
  const effectiveTargetDevice =
    availableTargetDevices.find((device) => device.id === requestedTargetDevice)?.id ??
    availableTargetDevices[0]?.id ??
    '__no-device__';
  const deviceOptions: ReadonlyArray<UiDropdownOption<string>> = availableTargetDevices.length
    ? availableTargetDevices.map((device) => ({
        value: device.id,
        label: device.label,
        icon:
          device.platform === hostPlatform ? <Monitor size={14} /> : <Smartphone size={14} />,
        disabled: device.disabled,
      }))
    : [{ value: '__no-device__', label: 'No devices available', icon: <Smartphone size={14} />, disabled: true }];

  const targetPlatformOptions: ReadonlyArray<UiDropdownOption<PlatformMenuValue>> = [
    ...projectTargetPlatforms.flatMap((platform) => {
      const option = platformOptions.find((candidate) => candidate.value === platform);
      return option ? [{ ...option, value: option.value as PlatformMenuValue }] : [];
    }),
    {
      value: '__platform-settings__',
      label: 'Platform Settings…',
      icon: <Settings2 size={14} />,
      separatorBefore: true,
      onSelect: () => {
        requestSettingsDialog('projectSettings');
        onCommand('settings.open');
      },
    },
  ];

  return (
    <section className="main-toolbar" aria-label="Editor toolbar">
      <div className="toolbar-left">
        <div className="ui-toolbar-group toolbar-group playback-group" aria-label="Playback controls">
          <UiIconButton
            active={runtimeState === 'running'}
            className="toolbar-button play"
            disabled={runtimeState === 'running' || runtimeState === 'faulted'}
            label={playLabel}
            onClick={() => onCommand('scene.play')}
          >
            <Play fill="currentColor" strokeWidth={0} size={14} />
          </UiIconButton>
          <UiIconButton
            active={runtimeState === 'paused'}
            className="toolbar-button"
            disabled={runtimeState !== 'running'}
            label="Pause"
            onClick={() => onCommand('scene.pause')}
          >
            <Pause size={14} />
          </UiIconButton>
          <UiIconButton
            className="toolbar-button"
            disabled={runtimeState === 'stopped'}
            label="Stop"
            onClick={() => onCommand('scene.stop')}
          >
            <Square size={13} />
          </UiIconButton>
          <UiIconButton
            className="toolbar-button"
            disabled={runtimeState !== 'paused'}
            label="Step"
            onClick={() => onCommand('scene.step')}
          >
            <StepForward size={14} />
          </UiIconButton>
          <ToolbarPlaybackOptions timeScale={timeScale} onTimeScaleChange={onTimeScaleChange} />
          <span
            className={`toolbar-runtime-state is-${runtimeState}`}
            data-testid="toolbar-runtime-state"
            title={runtimeError || runtimeLabel}
          >
            {runtimeLabel}
          </span>
        </div>
      </div>

      <div className="toolbar-center">
        <div className="ui-toolbar-group toolbar-group" aria-label="Transform mode">
          <UiDropdown
            ariaLabel="Transform origin"
            className="toolbar-origin-dropdown"
            disabled={authoringDisabled}
            onValueChange={setTransformOrigin}
            options={transformOriginOptions}
            value={transformOrigin}
          />
          <UiIconButton
            active={activeTool === 'select'}
            className="toolbar-button"
            label="Select (Q)"
            onClick={() => onCommand('viewport.select')}
          >
            <MousePointer2 size={14} />
          </UiIconButton>
          <UiIconButton
            active={activeTool === 'translate'}
            className="toolbar-button"
            disabled={authoringDisabled}
            label="Move (W)"
            onClick={() => onCommand('viewport.translate')}
          >
            <Move size={14} />
          </UiIconButton>
          <UiIconButton
            active={activeTool === 'rotate'}
            className="toolbar-button"
            disabled={authoringDisabled}
            label="Rotate (E)"
            onClick={() => onCommand('viewport.rotate')}
          >
            <Rotate3D size={14} />
          </UiIconButton>
          <UiIconButton
            active={activeTool === 'scale'}
            className="toolbar-button"
            disabled={authoringDisabled}
            label="Scale (R)"
            onClick={() => onCommand('viewport.scale')}
          >
            <Scaling size={14} />
          </UiIconButton>
          <UiIconButton
            active={activeTool === 'terrain'}
            className="toolbar-button"
            disabled={authoringDisabled || !terrainEnabled}
            label={terrainEnabled ? 'Terrain sculpt and paint' : 'Select a terrain to enable Terrain mode'}
            onClick={() => onCommand('viewport.terrain')}
          >
            <Mountain size={15} />
          </UiIconButton>
          <UiDropdown
            ariaLabel="Coordinate space"
            className="toolbar-coordinate-dropdown"
            disabled={authoringDisabled}
            onValueChange={(space) => onCoordinateSpaceChange?.(space)}
            options={coordinateSpaceOptions}
            value={coordinateSpace}
          />
        </div>

        <span className="toolbar-separator" />

        <div className="ui-toolbar-group toolbar-group" aria-label="Snapping controls">
          <ToolbarSnapMenu
            disabled={authoringDisabled}
            snapping={snapping}
            translationSnap={translationSnap}
            rotationSnap={rotationSnap}
            scaleSnap={scaleSnap}
            onToggleSnapping={onToggleSnapping}
            onTranslationSnapChange={onTranslationSnapChange}
            onRotationSnapChange={onRotationSnapChange}
            onScaleSnapChange={onScaleSnapChange}
          />
        </div>
      </div>

      <div className="toolbar-right">
        <UiDropdown
          ariaLabel="Target platform"
          className="toolbar-platform-dropdown"
          onValueChange={(platform) => {
            if (platform !== '__platform-settings__') onTargetPlatformChange?.(platform);
          }}
          options={targetPlatformOptions}
          value={effectiveTargetPlatform}
        />
        <UiDropdown
          ariaLabel="Target device"
          className="toolbar-device-dropdown"
          onValueChange={(deviceId) => {
            setLocalTargetDevice(deviceId);
            onTargetDeviceChange?.(deviceId);
          }}
          options={deviceOptions}
          value={effectiveTargetDevice}
        />
        <UiSplitButton
          ariaLabel="Build"
          className="toolbar-build-split"
          icon={<Hammer size={14} />}
          label="Build"
          menuAriaLabel="Build actions"
          onClick={() => onBuildAction?.('build')}
          onOptionSelect={(action) => onBuildAction?.(action)}
          options={buildOptions}
          variant="primary"
        />
      </div>
    </section>
  );
}
