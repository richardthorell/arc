import { Globe, Grid3X3, Mountain, Move, Pause, Play, Rotate3D, Scaling, Square, StepForward } from 'lucide-react';

import type { CommandId } from '../app/workbenchTypes';
import { UiButton, UiIconButton, UiSelectButton } from '../ui';

type MainToolbarProps = {
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
};

export function MainToolbar({ onCommand, activeTool = 'translate', coordinateSpace = 'world', snapping = false,
  translationSnap = 0.25, rotationSnap = 15, scaleSnap = 0.1, onToggleCoordinateSpace, onToggleSnapping,
  onCycleTranslationSnap, onCycleRotationSnap, onCycleScaleSnap, terrainEnabled = false,
  runtimeState = 'stopped', timeScale = 1, onCycleTimeScale }: MainToolbarProps) {
  return (
    <section className="main-toolbar" aria-label="Editor toolbar">
      <div className="toolbar-left">
        <div className="ui-toolbar-group toolbar-group playback-group" aria-label="Playback controls">
          <UiIconButton active={runtimeState === 'running'} className="toolbar-button play" label="Play" onClick={() => onCommand('scene.play')}>
            <Play fill="currentColor" strokeWidth={0} size={14} />
          </UiIconButton>
          <UiIconButton active={runtimeState === 'paused'} className="toolbar-button" label="Pause" onClick={() => onCommand('scene.pause')}>
            <Pause size={14} />
          </UiIconButton>
          <UiIconButton active={runtimeState === 'stopped'} className="toolbar-button" label="Stop" onClick={() => onCommand('scene.stop')}>
            <Square size={13} />
          </UiIconButton>
          <UiIconButton className="toolbar-button" label="Step" onClick={() => onCommand('scene.step')}>
            <StepForward size={14} />
          </UiIconButton>
          <UiSelectButton className="toolbar-select toolbar-select-narrow" onClick={onCycleTimeScale}
            title="Cycle preview simulation time scale">{timeScale}×</UiSelectButton>
        </div>
      </div>

      <div className="toolbar-center">
        <div className="ui-toolbar-group toolbar-group" aria-label="Transform mode">
          <UiSelectButton className="toolbar-select toolbar-select-compact">Pivot</UiSelectButton>
          <UiIconButton active={activeTool === 'translate'} className="toolbar-button" label="Move" onClick={() => onCommand('viewport.translate')}>
            <Move size={14} />
          </UiIconButton>
          <UiIconButton active={activeTool === 'rotate'} className="toolbar-button" label="Rotate" onClick={() => onCommand('viewport.rotate')}>
            <Rotate3D size={14} />
          </UiIconButton>
          <UiIconButton active={activeTool === 'scale'} className="toolbar-button" label="Scale" onClick={() => onCommand('viewport.scale')}>
            <Scaling size={14} />
          </UiIconButton>
          <UiIconButton active={activeTool === 'terrain'} className="toolbar-button" disabled={!terrainEnabled}
            label={terrainEnabled ? 'Terrain sculpt and paint' : 'Select a terrain to enable Terrain mode'}
            onClick={() => onCommand('viewport.terrain')}>
            <Mountain size={15} />
          </UiIconButton>
          <UiSelectButton active={coordinateSpace === 'local'} className="toolbar-select toolbar-select-compact"
            onClick={onToggleCoordinateSpace} title="Toggle world/local transform space">
            <Globe size={12} /> {coordinateSpace === 'world' ? 'World' : 'Local'}
          </UiSelectButton>
        </div>

        <span className="toolbar-separator" />

        <div className="ui-toolbar-group toolbar-group" aria-label="Snapping controls">
          <UiIconButton active={snapping} className="toolbar-button" label="Toggle transform snapping" onClick={onToggleSnapping}>
            <Grid3X3 size={14} />
          </UiIconButton>
          <UiSelectButton className="toolbar-select toolbar-select-narrow" onClick={onCycleRotationSnap}
            title="Cycle rotation snap increment">{rotationSnap}°</UiSelectButton>
          <UiSelectButton className="toolbar-select toolbar-select-narrow" onClick={onCycleTranslationSnap}
            title="Cycle translation snap increment">{translationSnap}</UiSelectButton>
          <UiSelectButton className="toolbar-select toolbar-select-narrow" onClick={onCycleScaleSnap}
            title="Cycle scale snap increment">{Math.round(scaleSnap * 100)}%</UiSelectButton>
        </div>
      </div>

      <div className="toolbar-right">
        <UiSelectButton className="toolbar-select toolbar-select-wide">Default Layout</UiSelectButton>
        <UiSelectButton className="toolbar-select">Windows</UiSelectButton>
        <UiButton className="toolbar-button build" variant="primary">
          Build
        </UiButton>
      </div>
    </section>
  );
}
