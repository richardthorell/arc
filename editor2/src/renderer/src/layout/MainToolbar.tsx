import { Globe, Grid3X3, Magnet, Move, Pause, Play, Rotate3D, Scaling, Square, StepForward } from 'lucide-react';

import type { CommandId } from '../app/workbenchTypes';
import { UiButton, UiIconButton, UiSelectButton } from '../ui';

type MainToolbarProps = {
  onCommand: (command: CommandId) => void;
};

export function MainToolbar({ onCommand }: MainToolbarProps) {
  return (
    <section className="main-toolbar" aria-label="Editor toolbar">
      <div className="toolbar-left">
        <div className="ui-toolbar-group toolbar-group playback-group" aria-label="Playback controls">
          <UiIconButton className="toolbar-button play" label="Play" onClick={() => onCommand('scene.play')}>
            <Play fill="currentColor" strokeWidth={0} size={14} />
          </UiIconButton>
          <UiIconButton className="toolbar-button" label="Pause" onClick={() => onCommand('scene.pause')}>
            <Pause size={14} />
          </UiIconButton>
          <UiIconButton className="toolbar-button" label="Stop" onClick={() => onCommand('scene.stop')}>
            <Square size={13} />
          </UiIconButton>
          <UiIconButton className="toolbar-button" label="Step" onClick={() => onCommand('scene.step')}>
            <StepForward size={14} />
          </UiIconButton>
        </div>
      </div>

      <div className="toolbar-center">
        <div className="ui-toolbar-group toolbar-group" aria-label="Transform mode">
          <UiSelectButton className="toolbar-select toolbar-select-compact">Pivot</UiSelectButton>
          <UiIconButton active className="toolbar-button selected" label="Move" onClick={() => onCommand('viewport.translate')}>
            <Move size={14} />
          </UiIconButton>
          <UiIconButton className="toolbar-button" label="Rotate" onClick={() => onCommand('viewport.rotate')}>
            <Rotate3D size={14} />
          </UiIconButton>
          <UiIconButton className="toolbar-button" label="Scale" onClick={() => onCommand('viewport.scale')}>
            <Scaling size={14} />
          </UiIconButton>
          <UiSelectButton className="toolbar-select toolbar-select-compact">
            <Globe size={12} /> Global
          </UiSelectButton>
        </div>

        <span className="toolbar-separator" />

        <div className="ui-toolbar-group toolbar-group" aria-label="Snapping controls">
          <UiIconButton className="toolbar-button" label="Grid snap">
            <Grid3X3 size={14} />
          </UiIconButton>
          <UiSelectButton className="toolbar-select toolbar-select-narrow">10 deg</UiSelectButton>
          <UiIconButton className="toolbar-button" label="Surface snap">
            <Magnet size={14} />
          </UiIconButton>
          <UiSelectButton className="toolbar-select toolbar-select-narrow">0.25</UiSelectButton>
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
