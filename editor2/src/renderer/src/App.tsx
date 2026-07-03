import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { Box, Cpu, Database, FolderTree, Gauge, Layers3, Play, Search, Settings, TerminalSquare } from 'lucide-react';

type StartupState = {
  appVersion: string;
  engineHostConnected: boolean;
  viewportMode: 'placeholder' | 'native' | 'streamed';
};

type Entity = {
  id: string;
  name: string;
  kind: string;
};

const entities: Entity[] = [
  { id: 'camera-main', name: 'Main Camera', kind: 'Camera' },
  { id: 'sun', name: 'Directional Light', kind: 'Light' },
  { id: 'environment', name: 'World Environment', kind: 'Environment' },
  { id: 'prototype-cube', name: 'Prototype Cube', kind: 'Mesh' },
  { id: 'floor', name: 'Floor Plane', kind: 'Mesh' },
];

const assets = ['materials/default.arcmat', 'meshes/cube.arcmesh', 'textures/checker.ktx', 'scenes/demo.arcscene'];

export function App() {
  const [startupState, setStartupState] = useState<StartupState | null>(null);
  const [selectedEntityId, setSelectedEntityId] = useState(entities[0].id);

  useEffect(() => {
    void window.arc.getStartupState().then(setStartupState);
  }, []);

  const selectedEntity = useMemo(
    () => entities.find((entity) => entity.id === selectedEntityId) ?? entities[0],
    [selectedEntityId],
  );

  return (
    <main className="app-shell">
      <header className="titlebar">
        <div className="brand-mark">arc</div>
        <nav className="main-nav">
          <button>File</button>
          <button>Edit</button>
          <button>Scene</button>
          <button>Render</button>
          <button>Tools</button>
        </nav>
        <div className="titlebar-spacer" />
        <div className="status-pill">editor2</div>
      </header>

      <section className="toolbar">
        <button className="primary-action">
          <Play size={14} />
          Play
        </button>
        <button>
          <Layers3 size={14} />
          Shaded
        </button>
        <button>
          <Gauge size={14} />
          Profiler
        </button>
        <div className="toolbar-spacer" />
        <label className="search-box">
          <Search size={14} />
          <input placeholder="Search scene, assets, commands" />
        </label>
      </section>

      <section className="workspace">
        <aside className="panel scene-panel">
          <PanelHeader icon={<FolderTree size={16} />} title="Scene" />
          <div className="entity-tree">
            {entities.map((entity) => (
              <button
                key={entity.id}
                className={entity.id === selectedEntityId ? 'entity-row selected' : 'entity-row'}
                onClick={() => setSelectedEntityId(entity.id)}
              >
                <Box size={14} />
                <span>{entity.name}</span>
                <small>{entity.kind}</small>
              </button>
            ))}
          </div>
        </aside>

        <section className="center-stack">
          <section className="viewport-panel">
            <div className="viewport-toolbar">
              <span>Viewport</span>
              <div>
                <button>Perspective</button>
                <button>Lit</button>
                <button>Gizmos</button>
              </div>
            </div>
            <div className="viewport-placeholder">
              <div className="viewport-grid" />
              <div className="viewport-message">
                <Cpu size={42} />
                <h2>Engine viewport placeholder</h2>
                <p>Phase 1 establishes the Electron shell. The native host and rendered viewport connect in the next phases.</p>
              </div>
            </div>
          </section>

          <section className="bottom-dock">
            <div className="panel console-panel">
              <PanelHeader icon={<TerminalSquare size={16} />} title="Console" />
              <code>[editor2] Electron shell initialized</code>
              <code>[editor2] Engine host connected: {String(startupState?.engineHostConnected ?? false)}</code>
              <code>[editor2] Viewport mode: {startupState?.viewportMode ?? 'placeholder'}</code>
            </div>
            <div className="panel assets-panel">
              <PanelHeader icon={<Database size={16} />} title="Assets" />
              {assets.map((asset) => (
                <button className="asset-row" key={asset}>{asset}</button>
              ))}
            </div>
          </section>
        </section>

        <aside className="panel inspector-panel">
          <PanelHeader icon={<Settings size={16} />} title="Inspector" />
          <section className="inspector-card">
            <label>Name</label>
            <input value={selectedEntity.name} readOnly />
          </section>
          <section className="inspector-card">
            <label>Kind</label>
            <input value={selectedEntity.kind} readOnly />
          </section>
          <section className="inspector-card transform-card">
            <label>Transform</label>
            <div className="triple-field"><span>X</span><input value="0.00" readOnly /></div>
            <div className="triple-field"><span>Y</span><input value="0.00" readOnly /></div>
            <div className="triple-field"><span>Z</span><input value="0.00" readOnly /></div>
          </section>
        </aside>
      </section>

      <footer className="statusbar">
        <span>arc editor2</span>
        <span>app {startupState?.appVersion ?? '...'}</span>
        <span>host disconnected</span>
      </footer>
    </main>
  );
}

function PanelHeader({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <header className="panel-header">
      {icon}
      <span>{title}</span>
    </header>
  );
}
