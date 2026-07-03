import { Box, ChevronDown, Grid3X3, Move3D, Pause, Play, RotateCw, Square, StepForward } from 'lucide-react';

import type { CommandId } from '../app/workbenchTypes';

type MainToolbarProps = {
  onCommand: (command: CommandId) => void;
};

export function MainToolbar({ onCommand }: MainToolbarProps) {
  return (
    <section className="main-toolbar" aria-label="Editor toolbar">
      <div className="toolbar-left">
        <div className="toolbar-group playback-group" aria-label="Playback controls">
          <button className="toolbar-button play" title="Play" onClick={() => onCommand('scene.play')}><Play size={14} /></button>
          <button className="toolbar-button" title="Pause" onClick={() => onCommand('scene.pause')}><Pause size={14} /></button>
          <button className="toolbar-button" title="Stop" onClick={() => onCommand('scene.stop')}><Square size={13} /></button>
          <button className="toolbar-button" title="Step" onClick={() => onCommand('scene.step')}><StepForward size={14} /></button>
        </div>

        <span className="toolbar-separator" />

        <div className="toolbar-group" aria-label="Transform mode">
          <button className="toolbar-select toolbar-select-compact">Pivot <ChevronDown size={12} /></button>
          <button className="toolbar-button selected" title="Move" onClick={() => onCommand('viewport.translate')}><Move3D size={15} /></button>
          <button className="toolbar-button" title="Rotate" onClick={() => onCommand('viewport.rotate')}><RotateCw size={14} /></button>
          <button className="toolbar-button" title="Scale" onClick={() => onCommand('viewport.scale')}><Box size={14} /></button>
          <button className="toolbar-select toolbar-select-compact">Global <ChevronDown size={12} /></button>
        </div>

        <span className="toolbar-separator" />

        <div className="toolbar-group" aria-label="Snapping controls">
          <button className="toolbar-button" title="Grid snap"><Grid3X3 size={14} /></button>
          <button className="toolbar-select toolbar-select-narrow">10° <ChevronDown size={12} /></button>
          <button className="toolbar-button" title="Surface snap">⌘</button>
          <button className="toolbar-select toolbar-select-narrow">0.25 <ChevronDown size={12} /></button>
        </div>
      </div>

      <div className="toolbar-right">
        <button className="toolbar-select toolbar-select-wide">Default Layout <ChevronDown size={12} /></button>
        <button className="toolbar-select">Windows <ChevronDown size={12} /></button>
        <button className="toolbar-button build">Build</button>
      </div>
    </section>
  );
}
