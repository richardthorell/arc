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
  Mountain,
  MousePointer2,
  Move,
  Pause,
  Play,
  RefreshCw,
  Rotate3D,
  Scaling,
  Settings2,
  Square,
  StepForward,
  Trash2,
} from 'lucide-react';

import type { CommandId } from '../app/workbenchTypes';
import {
  UiButton,
  UiDropdown,
  UiIconButton,
  UiSelectButton,
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

const transformOriginOptions: ReadonlyArray<UiDropdownOption<ToolbarTransformOrigin>> = [
  { value: 'pivot', label: 'Pivot', icon: <Crosshair size={12} /> },
  { value: 'center', label: 'Center', icon: <CircleDot size={12} /> },
];

const coordinateSpaceOptions: ReadonlyArray<UiDropdownOption<ToolbarCoordinateSpace>> = [
  { value: 'world', label: 'World', icon: <Globe size={12} /> },
  { value: 'local', label: 'Local', icon: <Box size={12} /> },
];

const translationSnapOptions: ReadonlyArray<UiDropdownOption<string>> = [
  0.01, 0.05, 0.1, 0.25, 0.5, 1, 5, 10,
].map((value) => ({ value: String(value), label: String(value) }));

const rotationSnapOptions: ReadonlyArray<UiDropdownOption<string>> = [1, 5, 10, 15, 30, 45, 90].map((value) => ({
  value: String(value),
  label: `${value}°`,
}));

const scaleSnapOptions: ReadonlyArray<UiDropdownOption<string>> = [0.01, 0.05, 0.1, 0.25, 0.5, 1].map((value) => ({
  value: String(value),
  label: `${Math.round(value * 100)}%`,
}));

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

  return (
    <span className="toolbar-snap-menu" ref={rootRef}>
      <UiButton
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label="Snap settings"
        className={`toolbar-snap-trigger${snapping ? ' is-active' : ''}`}
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
            <span className="toolbar-snap-check" aria-hidden="true">{snapping ? <Check size={13} /> : null}</span>
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
  runtimeState?: 'stopped' | 'running' | 'paused' | 'faulted';
  timeScale?: number;
  onCycleTimeScale?: () => void;
  targetPlatform?: EditorTargetPlatform;
  onTargetPlatformChange?: (platform: EditorTargetPlatform) => void;
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
  timeScale = 1,
  onCycleTimeScale,
  targetPlatform = 'windows',
  onTargetPlatformChange,
  onBuildAction,
}: MainToolbarProps) {
  const [transformOrigin, setTransformOrigin] = useState<ToolbarTransformOrigin>('pivot');

  return (
    <section className="main-toolbar" aria-label="Editor toolbar">
      <div className="toolbar-left">
        <div className="ui-toolbar-group toolbar-group playback-group" aria-label="Playback controls">
          <UiIconButton
            active={runtimeState === 'running'}
            className="toolbar-button play"
            label="Play"
            onClick={() => onCommand('scene.play')}
          >
            <Play fill="currentColor" strokeWidth={0} size={14} />
          </UiIconButton>
          <UiIconButton
            active={runtimeState === 'paused'}
            className="toolbar-button"
            label="Pause"
            onClick={() => onCommand('scene.pause')}
          >
            <Pause size={14} />
          </UiIconButton>
          <UiIconButton
            active={runtimeState === 'stopped'}
            className="toolbar-button"
            label="Stop"
            onClick={() => onCommand('scene.stop')}
          >
            <Square size={13} />
          </UiIconButton>
          <UiIconButton className="toolbar-button" label="Step" onClick={() => onCommand('scene.step')}>
            <StepForward size={14} />
          </UiIconButton>
          <UiSelectButton
            className="toolbar-select toolbar-select-narrow"
            onClick={onCycleTimeScale}
            title="Cycle preview simulation time scale"
          >
            {timeScale}×
          </UiSelectButton>
        </div>
      </div>

      <div className="toolbar-center">
        <div className="ui-toolbar-group toolbar-group" aria-label="Transform mode">
          <UiDropdown
            ariaLabel="Transform origin"
            className="toolbar-origin-dropdown"
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
            label="Move (W)"
            onClick={() => onCommand('viewport.translate')}
          >
            <Move size={14} />
          </UiIconButton>
          <UiIconButton
            active={activeTool === 'rotate'}
            className="toolbar-button"
            label="Rotate (E)"
            onClick={() => onCommand('viewport.rotate')}
          >
            <Rotate3D size={14} />
          </UiIconButton>
          <UiIconButton
            active={activeTool === 'scale'}
            className="toolbar-button"
            label="Scale (R)"
            onClick={() => onCommand('viewport.scale')}
          >
            <Scaling size={14} />
          </UiIconButton>
          <UiIconButton
            active={activeTool === 'terrain'}
            className="toolbar-button"
            disabled={!terrainEnabled}
            label={terrainEnabled ? 'Terrain sculpt and paint' : 'Select a terrain to enable Terrain mode'}
            onClick={() => onCommand('viewport.terrain')}
          >
            <Mountain size={15} />
          </UiIconButton>
          <UiDropdown
            ariaLabel="Coordinate space"
            className="toolbar-coordinate-dropdown"
            onValueChange={(space) => onCoordinateSpaceChange?.(space)}
            options={coordinateSpaceOptions}
            value={coordinateSpace}
          />
        </div>

        <span className="toolbar-separator" />

        <div className="ui-toolbar-group toolbar-group" aria-label="Snapping controls">
          <ToolbarSnapMenu
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
          onValueChange={(platform) => onTargetPlatformChange?.(platform)}
          options={platformOptions}
          value={targetPlatform}
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
