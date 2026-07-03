import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import {
  AlertTriangle,
  Box,
  Bug,
  ChevronDown,
  ChevronRight,
  Circle,
  Code2,
  Cpu,
  Database,
  FileCode2,
  FileText,
  Folder,
  FolderTree,
  Gauge,
  GitBranch,
  Layers3,
  Lightbulb,
  Package,
  Play,
  Search,
  Settings,
  SlidersHorizontal,
  Sparkles,
  TerminalSquare,
} from 'lucide-react';

import { flattenScene, mockHost } from './services/mockHost';
import type { AssetItem, ConsoleEvent, ProjectSnapshot, SceneEntity } from './services/mockHost';

type StartupState = {
  appVersion: string;
  engineHostConnected: boolean;
  viewportMode: 'placeholder' | 'native' | 'streamed';
};

type ActivityView = 'explorer' | 'search' | 'render' | 'debug' | 'extensions';
type BottomPanel = 'problems' | 'output' | 'debugConsole' | 'terminal' | 'profiler';

export function App() {
  const [startupState, setStartupState] = useState<StartupState | null>(null);
  const [project, setProject] = useState<ProjectSnapshot | null>(null);
  const [selectedEntityId, setSelectedEntityId] = useState('camera-main');
  const [activeView, setActiveView] = useState<ActivityView>('explorer');
  const [bottomPanel, setBottomPanel] = useState<BottomPanel>('output');
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>('asset-scene-demo');
  const [lastCommand, setLastCommand] = useState('Workbench ready');

  useEffect(() => {
    void window.arc.getStartupState().then(setStartupState);
    void mockHost.getProjectSnapshot().then(setProject);
  }, []);

  const selectedEntity = useMemo(() => {
    if (!project) {
      return null;
    }
    return flattenScene(project.scene).find((entity) => entity.id === selectedEntityId) ?? null;
  }, [project, selectedEntityId]);

  const selectedAsset = useMemo(
    () => project?.assets.find((asset) => asset.id === selectedAssetId) ?? null,
    [project, selectedAssetId],
  );

  const runNoop = async (command: string) => {
    const result = await mockHost.executeNoop(command);
    setLastCommand(result.succeeded ? `${result.command} queued` : `${result.command} failed`);
  };

  const selectEntity = async (entityId: string) => {
    const result = await mockHost.selectEntity(entityId);
    setSelectedEntityId(result.selectedEntityId);
  };

  return (
    <main className="workbench-shell">
      <header className="workbench-titlebar">
        <div className="traffic-spacer" />
        <nav className="menu-bar">
          <button>File</button>
          <button>Edit</button>
          <button>Selection</button>
          <button>View</button>
          <button>Scene</button>
          <button>Render</button>
          <button>Tools</button>
        </nav>
        <label className="command-center">
          <Search size={14} />
          <input placeholder="Search commands, assets, entities" />
        </label>
        <div className="window-title">arc editor2</div>
      </header>

      <section className="workbench-body">
        <aside className="activity-bar">
          <ActivityButton active={activeView === 'explorer'} title="Explorer" onClick={() => setActiveView('explorer')}>
            <FolderTree size={22} />
          </ActivityButton>
          <ActivityButton active={activeView === 'search'} title="Search" onClick={() => setActiveView('search')}>
            <Search size={22} />
          </ActivityButton>
          <ActivityButton active={activeView === 'render'} title="Render" onClick={() => setActiveView('render')}>
            <Layers3 size={22} />
          </ActivityButton>
          <ActivityButton active={activeView === 'debug'} title="Debug" onClick={() => setActiveView('debug')}>
            <Bug size={22} />
          </ActivityButton>
          <div className="activity-spacer" />
          <ActivityButton active={activeView === 'extensions'} title="Extensions" onClick={() => setActiveView('extensions')}>
            <Package size={22} />
          </ActivityButton>
          <ActivityButton active={false} title="Settings" onClick={() => void runNoop('open-settings')}>
            <Settings size={22} />
          </ActivityButton>
        </aside>

        <aside className="side-bar">
          <SideBarContent
            activeView={activeView}
            project={project}
            selectedEntityId={selectedEntityId}
            selectedAssetId={selectedAssetId}
            onSelectEntity={selectEntity}
            onSelectAsset={setSelectedAssetId}
          />
        </aside>

        <section className="editor-region">
          <section className="editor-tabs">
            <button className="tab active"><FileText size={14} /> demo.arcscene</button>
            <button className="tab"><Layers3 size={14} /> Render Graph</button>
            <button className="tab"><FileCode2 size={14} /> pbr_lit.hlsl</button>
          </section>

          <section className="scene-editor">
            <div className="viewport-toolbar">
              <div className="viewport-toolbar-group">
                <button onClick={() => void runNoop('play-scene')}><Play size={14} /> Play</button>
                <button onClick={() => void runNoop('build-lighting')}><Lightbulb size={14} /> Lighting</button>
                <button onClick={() => void runNoop('open-render-options')}><SlidersHorizontal size={14} /> Render Options</button>
              </div>
              <div className="viewport-toolbar-group muted">
                <span>Perspective</span>
                <span>Lit</span>
                <span>Gizmos On</span>
              </div>
            </div>

            <div className="viewport-canvas">
              <div className="viewport-grid" />
              <div className="axis-gizmo"><span>X</span><span>Y</span><span>Z</span></div>
              <div className="viewport-card">
                <Sparkles size={44} />
                <h2>Scene viewport</h2>
                <p>Mock workbench mode. The UI is wired through the same shape we will use for the native host.</p>
                <div className="viewport-stats-grid">
                  <Stat label="FPS" value={project?.renderStats.fps ?? 0} />
                  <Stat label="Frame" value={`${project?.renderStats.frameTimeMs ?? 0} ms`} />
                  <Stat label="Draws" value={project?.renderStats.drawCalls ?? 0} />
                  <Stat label="Tris" value={project?.renderStats.triangles.toLocaleString() ?? '0'} />
                </div>
              </div>
            </div>
          </section>

          <section className="bottom-panel">
            <div className="bottom-tabs">
              <BottomTab active={bottomPanel === 'problems'} onClick={() => setBottomPanel('problems')}>Problems</BottomTab>
              <BottomTab active={bottomPanel === 'output'} onClick={() => setBottomPanel('output')}>Output</BottomTab>
              <BottomTab active={bottomPanel === 'debugConsole'} onClick={() => setBottomPanel('debugConsole')}>Debug Console</BottomTab>
              <BottomTab active={bottomPanel === 'terminal'} onClick={() => setBottomPanel('terminal')}>Terminal</BottomTab>
              <BottomTab active={bottomPanel === 'profiler'} onClick={() => setBottomPanel('profiler')}>Profiler</BottomTab>
            </div>
            <BottomPanelContent panel={bottomPanel} events={project?.console ?? []} lastCommand={lastCommand} project={project} />
          </section>
        </section>

        <aside className="inspector-bar">
          <Inspector entity={selectedEntity} asset={selectedAsset} />
        </aside>
      </section>

      <footer className="status-bar">
        <span><GitBranch size={13} /> main</span>
        <span><Circle size={10} /> {startupState?.engineHostConnected ? 'host connected' : 'mock host'}</span>
        <span>{project?.activeScene ?? 'no scene'}</span>
        <span className="status-spacer" />
        <span>{lastCommand}</span>
        <span>editor {startupState?.appVersion ?? '...'}</span>
      </footer>
    </main>
  );
}

function ActivityButton({ active, title, onClick, children }: { active: boolean; title: string; onClick: () => void; children: ReactNode }) {
  return <button className={active ? 'activity-button active' : 'activity-button'} title={title} onClick={onClick}>{children}</button>;
}

function SideBarContent({ activeView, project, selectedEntityId, selectedAssetId, onSelectEntity, onSelectAsset }: {
  activeView: ActivityView;
  project: ProjectSnapshot | null;
  selectedEntityId: string;
  selectedAssetId: string | null;
  onSelectEntity: (entityId: string) => void;
  onSelectAsset: (assetId: string) => void;
}) {
  if (!project) {
    return <div className="side-loading">Loading workbench data...</div>;
  }

  if (activeView === 'search') {
    return <Panel title="Search"><input className="panel-search" placeholder="Search scene and assets" /><EmptyState icon={<Search />} title="No query yet" text="Search is mocked until the native host is connected." /></Panel>;
  }

  if (activeView === 'render') {
    return (
      <Panel title="Render">
        <PropertyRow label="Backend" value="Vulkan placeholder" />
        <PropertyRow label="Render Graph" value="Depth / GBuffer / Lighting" />
        <PropertyRow label="Visible" value={String(project.renderStats.visibleEntities)} />
        <PropertyRow label="GPU Memory" value={`${project.renderStats.gpuMemoryMb} MB`} />
      </Panel>
    );
  }

  if (activeView === 'debug') {
    return <Panel title="Run and Debug"><EmptyState icon={<Bug />} title="No active session" text="Play/debug commands are no-ops for now." /></Panel>;
  }

  if (activeView === 'extensions') {
    return <Panel title="Extensions"><EmptyState icon={<Package />} title="Extension host later" text="Reserved for importers, tools, and editor plugins." /></Panel>;
  }

  return (
    <div className="explorer-view">
      <Panel title={project.name.toUpperCase()}>
        <TreeSection title="Scene Hierarchy">
          {project.scene.map((entity) => (
            <SceneTreeItem key={entity.id} entity={entity} depth={0} selectedEntityId={selectedEntityId} onSelectEntity={onSelectEntity} />
          ))}
        </TreeSection>
        <TreeSection title="Assets">
          {project.assets.map((asset) => (
            <AssetRow key={asset.id} asset={asset} selected={asset.id === selectedAssetId} onSelect={() => onSelectAsset(asset.id)} />
          ))}
        </TreeSection>
      </Panel>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return <section className="workbench-panel"><header>{title}</header><div>{children}</div></section>;
}

function TreeSection({ title, children }: { title: string; children: ReactNode }) {
  return <section className="tree-section"><h3><ChevronDown size={14} /> {title}</h3>{children}</section>;
}

function SceneTreeItem({ entity, depth, selectedEntityId, onSelectEntity }: {
  entity: SceneEntity;
  depth: number;
  selectedEntityId: string;
  onSelectEntity: (entityId: string) => void;
}) {
  const hasChildren = Boolean(entity.children?.length);
  return (
    <div>
      <button
        className={entity.id === selectedEntityId ? 'tree-row selected' : 'tree-row'}
        style={{ paddingLeft: 10 + depth * 14 }}
        onClick={() => onSelectEntity(entity.id)}
      >
        {hasChildren ? <ChevronDown size={13} /> : <ChevronRight size={13} className="ghost" />}
        <EntityIcon kind={entity.kind} />
        <span>{entity.name}</span>
        {!entity.active && <small>off</small>}
      </button>
      {entity.children?.map((child) => (
        <SceneTreeItem key={child.id} entity={child} depth={depth + 1} selectedEntityId={selectedEntityId} onSelectEntity={onSelectEntity} />
      ))}
    </div>
  );
}

function EntityIcon({ kind }: { kind: SceneEntity['kind'] }) {
  if (kind === 'camera') return <Code2 size={14} />;
  if (kind === 'light') return <Lightbulb size={14} />;
  if (kind === 'folder') return <Folder size={14} />;
  return <Box size={14} />;
}

function AssetRow({ asset, selected, onSelect }: { asset: AssetItem; selected: boolean; onSelect: () => void }) {
  return <button className={selected ? 'tree-row selected' : 'tree-row'} onClick={onSelect}><AssetIcon kind={asset.kind} /><span>{asset.name}</span><small>{asset.status}</small></button>;
}

function AssetIcon({ kind }: { kind: AssetItem['kind'] }) {
  if (kind === 'scene') return <FileText size={14} />;
  if (kind === 'shader') return <FileCode2 size={14} />;
  if (kind === 'folder') return <Folder size={14} />;
  return <Database size={14} />;
}

function Inspector({ entity, asset }: { entity: SceneEntity | null; asset: AssetItem | null }) {
  return (
    <section className="inspector">
      <header>Inspector</header>
      {entity ? (
        <>
          <h2>{entity.name}</h2>
          <PropertyRow label="Kind" value={entity.kind} />
          <PropertyRow label="Active" value={entity.active ? 'true' : 'false'} />
          <VectorEditor title="Position" value={entity.transform.position} />
          <VectorEditor title="Rotation" value={entity.transform.rotation} />
          <VectorEditor title="Scale" value={entity.transform.scale} />
          <section className="component-list"><h3>Components</h3>{entity.components.length ? entity.components.map((component) => <button key={component}>{component}</button>) : <span>No components</span>}</section>
        </>
      ) : <EmptyState icon={<Settings />} title="Nothing selected" text="Select an entity or asset." />}
      {asset && <section className="asset-inspector"><h3>Selected Asset</h3><PropertyRow label="Name" value={asset.name} /><PropertyRow label="Path" value={asset.path} /><PropertyRow label="Status" value={asset.status} /></section>}
    </section>
  );
}

function VectorEditor({ title, value }: { title: string; value: { x: number; y: number; z: number } }) {
  return <section className="vector-editor"><h3>{title}</h3><NumberField label="X" value={value.x} /><NumberField label="Y" value={value.y} /><NumberField label="Z" value={value.z} /></section>;
}

function NumberField({ label, value }: { label: string; value: number }) {
  return <label className="number-field"><span>{label}</span><input value={Number(value).toFixed(2)} readOnly /></label>;
}

function PropertyRow({ label, value }: { label: string; value: ReactNode }) {
  return <div className="property-row"><span>{label}</span><strong>{value}</strong></div>;
}

function BottomTab({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return <button className={active ? 'bottom-tab active' : 'bottom-tab'} onClick={onClick}>{children}</button>;
}

function BottomPanelContent({ panel, events, lastCommand, project }: { panel: BottomPanel; events: ConsoleEvent[]; lastCommand: string; project: ProjectSnapshot | null }) {
  if (panel === 'problems') {
    return <div className="bottom-content"><LogLine level="warning" text="1 mock warning: shader import is still running." /></div>;
  }
  if (panel === 'profiler') {
    return <div className="bottom-content profiler-grid"><Stat label="FPS" value={project?.renderStats.fps ?? 0} /><Stat label="Frame" value={`${project?.renderStats.frameTimeMs ?? 0} ms`} /><Stat label="Draw Calls" value={project?.renderStats.drawCalls ?? 0} /><Stat label="GPU MB" value={project?.renderStats.gpuMemoryMb ?? 0} /></div>;
  }
  if (panel === 'terminal') {
    return <div className="bottom-content terminal"><code>$ arc editor2 --mock-host</code><code>last: {lastCommand}</code></div>;
  }
  return <div className="bottom-content">{events.map((event) => <LogLine key={event.id} level={event.level} text={`[${event.timestamp}] [${event.source}] ${event.message}`} />)}</div>;
}

function LogLine({ level, text }: { level: ConsoleEvent['level'] | 'warning'; text: string }) {
  return <code className={`log-line ${level}`}>{level === 'warning' && <AlertTriangle size={13} />} {text}</code>;
}

function Stat({ label, value }: { label: string; value: ReactNode }) {
  return <div className="stat"><span>{label}</span><strong>{value}</strong></div>;
}

function EmptyState({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return <div className="empty-state">{icon}<h3>{title}</h3><p>{text}</p></div>;
}
