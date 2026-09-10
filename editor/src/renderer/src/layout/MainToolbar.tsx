import {
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

const buildOptions: ReadonlyArray<UiSplitButtonOption<ToolbarBuildAction>> = [
  { value: 'build', label: 'Build', icon: <Hammer size={14} /> },
  { value: 'rebuild', label: 'Rebuild', icon: <RefreshCw size={14} /> },
  { value: 'configure', label: 'Configure', icon: <Settings2 size={14} /> },
  { value: 'clean', label: 'Clean', icon: <Trash2 size={14} /> },
];

export type MainToolbarProps = {
  onCommand: (command: CommandId) => void;
  activeTool?: 'select' | 'translate' | 'rotate' | 'scale' | 'terrain';
  terrainEnabled?: boolean;
  coordinateSpace?: 'world' | 'local';
  snapping?: boolean;
  translationSnap?: number;
  rotationSnap?: number;
  scaleSnap?: number;
  onToggleCoordinateSpace?: () => void;
  onToggleSnapping?: () => void;
  onCycleTranslationSnap?: () => void;
  onCycleRotationSnap?: () => void;
  onCycleScaleSnap?: () => void;
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
  onToggleCoordinateSpace,
  onToggleSnapping,
  onCycleTranslationSnap,
  onCycleRotationSnap,
  onCycleScaleSnap,
  terrainEnabled = false,
  runtimeState = 'stopped',
  timeScale = 1,
  onCycleTimeScale,
  targetPlatform = 'windows',
  onTargetPlatformChange,
  onBuildAction,
}: MainToolbarProps) {
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
          <UiSelectButton className="toolbar-select toolbar-select-compact">Pivot</UiSelectButton>
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
          <UiSelectButton
            active={coordinateSpace === 'local'}
            className="toolbar-select toolbar-select-compact"
            onClick={onToggleCoordinateSpace}
            title="Toggle world/local transform space"
          >
            <Globe size={12} /> {coordinateSpace === 'world' ? 'World' : 'Local'}
          </UiSelectButton>
        </div>

        <span className="toolbar-separator" />

        <div className="ui-toolbar-group toolbar-group" aria-label="Snapping controls">
          <UiIconButton
            active={snapping}
            className="toolbar-button"
            label="Toggle transform snapping"
            onClick={onToggleSnapping}
          >
            <Grid3X3 size={14} />
          </UiIconButton>
          <UiSelectButton
            className="toolbar-select toolbar-select-narrow"
            onClick={onCycleRotationSnap}
            title="Cycle rotation snap increment"
          >
            {rotationSnap}°
          </UiSelectButton>
          <UiSelectButton
            className="toolbar-select toolbar-select-narrow"
            onClick={onCycleTranslationSnap}
            title="Cycle translation snap increment"
          >
            {translationSnap}
          </UiSelectButton>
          <UiSelectButton
            className="toolbar-select toolbar-select-narrow"
            onClick={onCycleScaleSnap}
            title="Cycle scale snap increment"
          >
            {Math.round(scaleSnap * 100)}%
          </UiSelectButton>
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
