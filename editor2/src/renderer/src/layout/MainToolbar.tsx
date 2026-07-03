import { Box, Move3D, Pause, Play, RotateCw, Square, StepForward } from 'lucide-react';

import type { CommandId } from '../app/workbenchTypes';

type MainToolbarProps = {
  onCommand: (command: CommandId) => void;
};

export function MainToolbar({ onCommand }: MainToolbarProps) {
  return (
    <section className="main-toolbar">
      <div className="toolbar-group playback-group">
        <button className="toolbar-button play" onClick={() => onCommand('scene.play')}><Play size={14} /></button>
        <button className="toolbar-button" onClick={() => onCommand('scene.pause')}><Pause size={14} /></button>
        <button className="toolbar-button" onClick={() => onCommand('scene.stop')}><Square size={13} /></button>
        <button className="toolbar-button" onClick={() => onCommand('scene.step')}><StepForward size={14} /></button>
      </div>
      <div className="toolbar-group">
        <button className="toolbar-button selected" onClick={() => onCommand('viewport.translate')}><Move3D size={14} /></button>
        <button className="toolbar-button" onClick={() => onCommand('viewport.rotate')}><RotateCw size={14} /></button>
        <button className="toolbar-button" onClick={() => onCommand('viewport.scale')}><Box size={14} /></button>
      </div>
      <div className="toolbar-group">
        <button className="toolbar-select">Pivot</button>
        <button className="toolbar-select">Global</button>
        <button className="toolbar-select">Grid 0.25</button>
        <button className="toolbar-select">10°</button>
      </div>
      <div className="toolbar-spacer" />
      <div className="toolbar-group">
        <button className="toolbar-select">Default Layout</button>
        <button className="toolbar-select">Windows</button>
        <button className="toolbar-button build">Build</button>
      </div>
    </section>
  );
}
