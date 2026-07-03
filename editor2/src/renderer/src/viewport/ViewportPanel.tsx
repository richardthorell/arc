import {
  Box,
  Camera,
  Crosshair,
  Eye,
  Focus,
  Grid3X3,
  Hand,
  Maximize2,
  MousePointer2,
  Move3D,
  Orbit,
  RotateCw,
  Scaling,
} from 'lucide-react';

import type { CommandId } from '../app/workbenchTypes';
import type { ProjectSnapshot } from '../services/mockHost';

import './viewport.css';

type ViewportPanelProps = {
  project: ProjectSnapshot | null;
  onCommand: (command: CommandId) => void;
};

export function ViewportPanel({ project, onCommand }: ViewportPanelProps) {
  const stats = project?.renderStats;

  return (
    <section className="arc-viewport-shell">
      <header className="arc-viewport-header">
        <div className="arc-viewport-title">
          <Camera size={14} />
          <span>Viewport 1</span>
        </div>
        <div className="arc-viewport-view-options">
          <button>Perspective</button>
          <button>Lit</button>
          <button>Show</button>
        </div>
        <div className="arc-viewport-header-spacer" />
        <div className="arc-viewport-view-options compact">
          <button title="Camera speed">Speed 4</button>
          <button title="Realtime"><Eye size={13} /></button>
          <button title="Maximize"><Maximize2 size={13} /></button>
        </div>
      </header>

      <div className="arc-viewport-body">
        <div className="arc-viewport-scene-bg" />
        <div className="arc-viewport-depth-fog" />
        <div className="arc-viewport-terrain" />
        <div className="arc-viewport-object-shadow" />
        <div className="arc-viewport-selected-object">
          <div className="arc-selected-roof" />
          <div className="arc-selected-body" />
          <div className="arc-selection-outline" />
        </div>
        <div className="arc-transform-gizmo" aria-label="Transform gizmo">
          <span className="axis-y" />
          <span className="axis-x" />
          <span className="axis-z" />
          <strong />
        </div>

        <aside className="arc-viewport-tool-strip">
          <button title="Select" onClick={() => onCommand('viewport.select')}><MousePointer2 size={16} /></button>
          <button title="Pan"><Hand size={16} /></button>
          <button className="active" title="Translate" onClick={() => onCommand('viewport.translate')}><Move3D size={16} /></button>
          <button title="Rotate" onClick={() => onCommand('viewport.rotate')}><RotateCw size={16} /></button>
          <button title="Scale" onClick={() => onCommand('viewport.scale')}><Scaling size={16} /></button>
          <button title="Frame selected" onClick={() => onCommand('viewport.frameSelected')}><Focus size={16} /></button>
        </aside>

        <aside className="arc-viewport-stats">
          <dl>
            <div><dt>FPS</dt><dd>{stats?.fps.toFixed(1) ?? '0.0'}</dd></div>
            <div><dt>Frame</dt><dd>{stats?.frameTimeMs.toFixed(2) ?? '0.00'} ms</dd></div>
            <div><dt>Draw Calls</dt><dd>{stats?.drawCalls.toLocaleString() ?? '0'}</dd></div>
            <div><dt>Triangles</dt><dd>{stats?.triangles.toLocaleString() ?? '0'}</dd></div>
            <div><dt>Objects</dt><dd>{stats?.visibleEntities ?? 0}</dd></div>
            <div><dt>Lights</dt><dd>{stats?.lights ?? 0}</dd></div>
          </dl>
        </aside>

        <div className="arc-viewport-overlay-top-left">
          <span><Grid3X3 size={13} /> Grid</span>
          <span><Crosshair size={13} /> Snapping 0.25</span>
          <span><Orbit size={13} /> Global</span>
        </div>

        <div className="arc-viewport-breadcrumb">
          <span>World</span>
          <span>Buildings</span>
          <span>Cabin_01</span>
          <strong>SM_Cabin</strong>
        </div>

        <div className="arc-axis-gizmo-large">
          <span className="axis-label-y">Y</span>
          <span className="axis-label-x">X</span>
          <span className="axis-label-z">Z</span>
        </div>

        <div className="arc-viewport-note">
          <Box size={18} />
          <span>Viewport shell only. Real engine rendering connects later.</span>
        </div>
      </div>
    </section>
  );
}
