import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import {
  AlertTriangle,
  Box,
  ChevronDown,
  ChevronRight,
  Code2,
  Database,
  FileCode2,
  FileText,
  Folder,
  Lightbulb,
  Package,
  Search,
  Settings,
} from 'lucide-react';

import { executeWorkbenchCommand } from './commandRegistry';
import { activityRegistry, dockPanelIds, getPanel } from './panelRegistry';
import { defaultWorkbenchLayout, useWorkbenchLayout } from './workbenchStore';
import type { ActivityId, CommandId, StartupState, WorkbenchPanelId } from './workbenchTypes';
import { ActivityBar } from '../layout/ActivityBar';
import { DockHost } from '../layout/DockHost';
import { MainToolbar } from '../layout/MainToolbar';
import { MenuBar } from '../layout/MenuBar';
import { StatusBar } from '../layout/StatusBar';
import { flattenScene, mockHost } from '../services/mockHost';
import type { AssetItem, ConsoleEvent, ProjectSnapshot, SceneEntity } from '../services/mockHost';
import { ViewportPanel } from '../viewport/ViewportPanel';

import './workbench.css';

const fallbackStartupState: StartupState = {
  appVersion: 'dev',
  engineHostConnected: false,
  viewportMode: 'placeholder',
};

export function Workbench() {
  const { layout, setLayout, resetLayout } = useWorkbenchLayout();
  const [startupState, setStartupState] = useState<StartupState | null>(null);
  const [project, setProject] = useState<ProjectSnapshot | null>(null);
  const [selectedEntityId, setSelectedEntityId] = useState('camera-main');
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>('asset-scene-demo');
  const [lastCommand, setLastCommand] = useState('Workbench ready');

  useEffect(() => {
    const startup = window.arc?.getStartupState?.() ?? Promise.resolve(fallbackStartupState);
    void startup.then(setStartupState).catch(() => setStartupState(fallbackStartupState));
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

  const runCommand = async (command: CommandId) => {
    if (command === 'layout.reset') {
      resetLayout();
      setLastCommand('Layout reset');
      return;
    }

    const result = await executeWorkbenchCommand(command);
    setLastCommand(result.message);
  };

  const selectEntity = async (entityId: string) => {
    const result = await mockHost.selectEntity(entityId);
    setSelectedEntityId(result.selectedEntityId);
  };

  const selectActivity = (activityId: ActivityId) => {
    const activity = activityRegistry.find((entry) => entry.id === activityId);
    const panel = activity ? getPanel(activity.panelId) : null;

    setLayout((current) => ({
      ...current,
      activeActivity: activityId,
      leftVisible: true,
      bottomVisible: panel?.defaultRegion === 'bottom' ? true : current.bottomVisible,
      activeCenterPanel: panel?.defaultRegion === 'center' ? panel.id : current.activeCenterPanel,
      activeBottomPanel: panel?.defaultRegion === 'bottom' ? panel.id : current.activeBottomPanel,
    }));
  };

  const setActiveCenterPanel = (panel: WorkbenchPanelId) => setLayout((current) => ({ ...current, activeCenterPanel: panel }));
  const setActiveRightPanel = (panel: WorkbenchPanelId) => setLayout((current) => ({ ...current, activeRightPanel: panel }));
  const setActiveBottomPanel = (panel: WorkbenchPanelId) => setLayout((current) => ({ ...current, activeBottomPanel: panel }));

  const renderLeftPanel = () => {
    if (!project) {
      return <div className="side-loading">Loading workbench data...</div>;
    }

    if (layout.activeActivity === 'scene') {
      return <ExplorerPanel project={project} selectedEntityId={selectedEntityId} selectedAssetId={selectedAssetId} onSelectEntity={selectEntity} onSelectAsset={setSelectedAssetId} />;
    }

    if (layout.activeActivity === 'assets') {
      return <AssetExplorerPanel project={project} selectedAssetId={selectedAssetId} onSelectAsset={setSelectedAssetId} />;
    }

    if (layout.activeActivity === 'search') {
      return <PlaceholderPanel icon={<Search />} title="Search" text="Search is mocked until the native host is connected." />;
    }

    if (layout.activeActivity === 'settings') {
      return <SettingsPanel onResetLayout={resetLayout} />;
    }

    const panelId = activityRegistry.find((entry) => entry.id === layout.activeActivity)?.panelId ?? 'hierarchy';
    const panel = getPanel(panelId);
    return <PlaceholderPanel icon={<panel.icon />} title={panel.title} text="This panel is available in the main dock area." />;
  };

  const renderCenterPanel = (panel: WorkbenchPanelId) => {
    if (panel === 'viewport') {
      return <ViewportPanel project={project} onCommand={runCommand} />;
    }

    if (panel === 'renderGraph') {
      return <RenderGraphPanel />;
    }

    return <ShaderEditorPanel />;
  };

  const renderRightPanel = (panel: WorkbenchPanelId) => {
    if (panel === 'inspector') {
      return <InspectorPanel entity={selectedEntity} asset={selectedAsset} />;
    }

    if (panel === 'lighting') {
      return <LightingPanel project={project} />;
    }

    return <WorldSettingsPanel project={project} />;
  };

  const renderBottomPanel = (panel: WorkbenchPanelId) => {
    if (panel === 'contentBrowser') {
      return <ContentBrowserPanel project={project} selectedAssetId={selectedAssetId} onSelectAsset={setSelectedAssetId} onCommand={runCommand} />;
    }

    if (panel === 'console') {
      return <ConsolePanel events={project?.console ?? []} lastCommand={lastCommand} />;
    }

    if (panel === 'versionControl') {
      return <VersionControlPanel onCommand={runCommand} />;
    }

    if (panel === 'aiAssistant') {
      return <AiAssistantPanel selectedEntity={selectedEntity} selectedAsset={selectedAsset} onCommand={runCommand} />;
    }

    return <ProfilerPanel project={project} />;
  };

  return (
    <main className="workbench-shell">
      <MenuBar projectTitle={`${project?.name ?? 'arc editor2'}${project ? '.arcscene*' : ''}`} onCommand={runCommand} />
      <MainToolbar onCommand={runCommand} />

      <section className="workbench-body workbench-body-foundation">
        <ActivityBar activeActivity={layout.activeActivity} onSelectActivity={selectActivity} />

        {layout.leftVisible && <aside className="side-bar">{renderLeftPanel()}</aside>}

        <section className="editor-region foundation-editor-region">
          <DockHost
            region="center"
            panelIds={dockPanelIds.center}
            activePanelId={layout.activeCenterPanel}
            onActivePanelChange={setActiveCenterPanel}
            renderPanel={renderCenterPanel}
          />

          {layout.bottomVisible && (
            <DockHost
              region="bottom"
              panelIds={dockPanelIds.bottom}
              activePanelId={layout.activeBottomPanel}
              onActivePanelChange={setActiveBottomPanel}
              renderPanel={renderBottomPanel}
            />
          )}
        </section>

        {layout.rightVisible && (
          <aside className="inspector-bar">
            <DockHost
              region="right"
              panelIds={dockPanelIds.right}
              activePanelId={layout.activeRightPanel}
              onActivePanelChange={setActiveRightPanel}
              renderPanel={renderRightPanel}
            />
          </aside>
        )}
      </section>

      <StatusBar startupState={startupState} activeScene={project?.activeScene} lastCommand={lastCommand} />
    </main>
  );
}

function ExplorerPanel({ project, selectedEntityId, selectedAssetId, onSelectEntity, onSelectAsset }: {
  project: ProjectSnapshot;
  selectedEntityId: string;
  selectedAssetId: string | null;
  onSelectEntity: (entityId: string) => void;
  onSelectAsset: (assetId: string) => void;
}) {
  return (
    <div className="explorer-view">
      <Panel title={project.name.toUpperCase()}>
        <TreeSection title="Scene Hierarchy">
          {project.scene.map((entity) => (
            <SceneTreeItem key={entity.id} entity={entity} depth={0} selectedEntityId={selectedEntityId} onSelectEntity={onSelectEntity} />
          ))}
        </TreeSection>
        <TreeSection title="Assets">
          {project.assets.slice(0, 7).map((asset) => (
            <AssetRow key={asset.id} asset={asset} selected={asset.id === selectedAssetId} onSelect={() => onSelectAsset(asset.id)} />
          ))}
        </TreeSection>
      </Panel>
    </div>
  );
}

function AssetExplorerPanel({ project, selectedAssetId, onSelectAsset }: {
  project: ProjectSnapshot;
  selectedAssetId: string | null;
  onSelectAsset: (assetId: string) => void;
}) {
  return (
    <Panel title="Assets">
      <TreeSection title="Project Assets">
        {project.assets.map((asset) => <AssetRow key={asset.id} asset={asset} selected={asset.id === selectedAssetId} onSelect={() => onSelectAsset(asset.id)} />)}
      </TreeSection>
    </Panel>
  );
}

function InspectorPanel({ entity, asset }: { entity: SceneEntity | null; asset: AssetItem | null }) {
  return (
    <section className="inspector foundation-inspector">
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
      ) : <PlaceholderPanel icon={<Settings />} title="Nothing selected" text="Select an entity or asset." />}
      {asset && <section className="asset-inspector"><h3>Selected Asset</h3><PropertyRow label="Name" value={asset.name} /><PropertyRow label="Path" value={asset.path} /><PropertyRow label="Status" value={asset.status} /></section>}
    </section>
  );
}

function ContentBrowserPanel({ project, selectedAssetId, onSelectAsset, onCommand }: {
  project: ProjectSnapshot | null;
  selectedAssetId: string | null;
  onSelectAsset: (assetId: string) => void;
  onCommand: (command: CommandId) => void;
}) {
  return (
    <section className="content-browser-foundation">
      <div className="content-browser-toolbar">
        <button onClick={() => onCommand('assets.import')}>+ Add</button>
        <button onClick={() => onCommand('assets.import')}>Import</button>
        <button onClick={() => onCommand('assets.saveAll')}>Save All</button>
        <span>Assets / Environment / Props</span>
      </div>
      <div className="asset-grid-foundation">
        {(project?.assets ?? []).map((asset) => <AssetCard key={asset.id} asset={asset} selected={asset.id === selectedAssetId} onSelect={() => onSelectAsset(asset.id)} />)}
      </div>
    </section>
  );
}

function ConsolePanel({ events, lastCommand }: { events: ConsoleEvent[]; lastCommand: string }) {
  return <div className="bottom-content">{events.map((event) => <LogLine key={event.id} level={event.level} text={`[${event.timestamp}] [${event.source}] ${event.message}`} />)}<LogLine level="debug" text={`last command: ${lastCommand}`} /></div>;
}

function VersionControlPanel({ onCommand }: { onCommand: (command: CommandId) => void }) {
  return (
    <section className="vcs-panel-foundation">
      <div className="property-row"><span>Branch</span><strong>main · ahead 3</strong></div>
      <TreeSection title="Changes">
        {['Assets/Environment/Props/Lantern.fbx', 'Scenes/MountainVillage.arcscene', 'Shaders/Water/WaterCommon.glsl'].map((file) => <button className="tree-row" key={file}><FileText size={14} /><span>{file}</span><small>M</small></button>)}
      </TreeSection>
      <textarea className="commit-message" defaultValue="Add cabin props and materials" />
      <div className="vcs-actions"><button onClick={() => onCommand('vcs.commit')}>Commit</button><button onClick={() => onCommand('vcs.pull')}>Pull</button><button onClick={() => onCommand('vcs.push')}>Push</button></div>
    </section>
  );
}

function AiAssistantPanel({ selectedEntity, selectedAsset, onCommand }: { selectedEntity: SceneEntity | null; selectedAsset: AssetItem | null; onCommand: (command: CommandId) => void }) {
  return (
    <section className="ai-panel-foundation">
      <header><strong>AI Assistant</strong><button onClick={() => onCommand('ai.newChat')}>+ New Chat</button></header>
      <p>You are in <strong>MountainVillage.arcscene</strong>.</p>
      <p>Selection: {selectedEntity?.name ?? selectedAsset?.name ?? 'none'}</p>
      <div className="ai-suggestion-grid"><button>Create material</button><button>Add point light</button><button>Find shader errors</button><button>Generate test scene</button></div>
      <label className="ai-input"><input placeholder="Ask arc anything..." /></label>
    </section>
  );
}

function ProfilerPanel({ project }: { project: ProjectSnapshot | null }) {
  return <div className="bottom-content profiler-grid"><Stat label="FPS" value={project?.renderStats.fps ?? 0} /><Stat label="Frame" value={`${project?.renderStats.frameTimeMs ?? 0} ms`} /><Stat label="Draw Calls" value={project?.renderStats.drawCalls ?? 0} /><Stat label="GPU MB" value={project?.renderStats.gpuMemoryMb ?? 0} /></div>;
}

function RenderGraphPanel() {
  return <PlaceholderPanel icon={<Package />} title="Render Graph" text="Depth Prepass → GBuffer → Lighting → Transparency → Post Process → UI Composite" />;
}

function ShaderEditorPanel() {
  return <PlaceholderPanel icon={<FileCode2 />} title="Shader Editor" text="Shader/source editing panel placeholder." />;
}

function LightingPanel({ project }: { project: ProjectSnapshot | null }) {
  return <PlaceholderPanel icon={<Lightbulb />} title="Lighting" text={`${project?.renderStats.lights ?? 0} lights in the current mock scene.`} />;
}

function WorldSettingsPanel({ project }: { project: ProjectSnapshot | null }) {
  return <PlaceholderPanel icon={<Settings />} title="World Settings" text={project?.activeScene ?? 'No active scene'} />;
}

function SettingsPanel({ onResetLayout }: { onResetLayout: () => void }) {
  return <Panel title="Settings"><button className="settings-action" onClick={onResetLayout}>Reset Layout</button><PropertyRow label="Theme" value="Arc Dark" /><PropertyRow label="Host" value="Mock" /></Panel>;
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
      <button className={entity.id === selectedEntityId ? 'tree-row selected' : 'tree-row'} style={{ paddingLeft: 10 + depth * 14 }} onClick={() => onSelectEntity(entity.id)}>
        {hasChildren ? <ChevronDown size={13} /> : <ChevronRight size={13} className="ghost" />}
        <EntityIcon kind={entity.kind} />
        <span>{entity.name}</span>
        {!entity.active && <small>off</small>}
      </button>
      {entity.children?.map((child) => <SceneTreeItem key={child.id} entity={child} depth={depth + 1} selectedEntityId={selectedEntityId} onSelectEntity={onSelectEntity} />)}
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

function AssetCard({ asset, selected, onSelect }: { asset: AssetItem; selected: boolean; onSelect: () => void }) {
  return <button className={selected ? 'asset-card-foundation selected' : 'asset-card-foundation'} onClick={onSelect}><AssetIcon kind={asset.kind} /><strong>{asset.name}</strong><span>{asset.kind}</span></button>;
}

function AssetIcon({ kind }: { kind: AssetItem['kind'] }) {
  if (kind === 'scene') return <FileText size={14} />;
  if (kind === 'shader') return <FileCode2 size={14} />;
  if (kind === 'folder') return <Folder size={14} />;
  return <Database size={14} />;
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

function LogLine({ level, text }: { level: ConsoleEvent['level'] | 'warning' | 'debug'; text: string }) {
  return <code className={`log-line ${level}`}>{level === 'warning' && <AlertTriangle size={13} />} {text}</code>;
}

function Stat({ label, value }: { label: string; value: ReactNode }) {
  return <div className="stat"><span>{label}</span><strong>{value}</strong></div>;
}

function PlaceholderPanel({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return <div className="empty-state">{icon}<h3>{title}</h3><p>{text}</p></div>;
}

void defaultWorkbenchLayout;
