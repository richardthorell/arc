import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties, PointerEvent as ReactPointerEvent, ReactNode } from 'react';
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
  FolderTree,
  MoreVertical,
  Lightbulb,
  Search,
  Settings,
  X,
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
import { UiButton, UiIconButton, UiPanel, UiTab, UiTabs, UiTextInput, UiTreeRow } from '../ui';
import { ViewportPanel } from '../viewport/ViewportPanel';
import { WorldEnvironmentInspector } from '../environment/WorldEnvironmentInspector';
import type { HostWorldEnvironment } from '../environment/environmentTypes';
import { InspectorPanel as DataDrivenInspector } from '../inspector/InspectorPanel';
import type { HostEntityId, HostResponse, InspectorEntitySnapshot } from '../inspector/inspectorTypes';
import { eulerDegreesToQuaternion, hostEntityKey, parseSelectedEntitySnapshot } from '../inspector/inspectorTypes';

import './workbench.css';

type HostSceneEntity = {
  entity: HostEntityId;
  name: string;
  kind: 'camera' | 'light' | 'environment' | 'mesh' | 'primitive' | 'imported' | 'unknown';
  active: boolean;
  selected: boolean;
};

type HostSceneSnapshot = {
  entities: HostSceneEntity[];
};

type HostAssetSnapshot = {
  path: string;
  kind: AssetItem['kind'] | 'environment' | 'unknown';
  imported: boolean;
  importRunning: boolean;
};

type HostProjectAssetsSnapshot = {
  projectName: string;
  projectRoot: string;
  assetRoot: string;
  assets: HostAssetSnapshot[];
};

type HostAssetThumbnailSnapshot = {
  path: string;
  width: number;
  height: number;
  dataUrl: string;
};

const fallbackStartupState: StartupState = {
  appVersion: 'dev',
  engineHostConnected: false,
  viewportMode: 'placeholder',
};

const sceneKindFromHost = (kind: HostSceneEntity['kind']): SceneEntity['kind'] => {
  if (kind === 'camera' || kind === 'light' || kind === 'environment' || kind === 'mesh') {
    return kind;
  }
  return 'mesh';
};

const assetKindFromHost = (kind: HostAssetSnapshot['kind']): AssetItem['kind'] => {
  if (kind === 'environment') return 'texture';
  if (kind === 'material' || kind === 'texture' || kind === 'shader' || kind === 'mesh' || kind === 'folder') {
    return kind;
  }
  return 'scene';
};

const assetNameFromPath = (value: string) => value.split(/[\\/]/).pop() || value;

const timestamp = () => new Date().toLocaleTimeString([], { hour12: false });

const sceneRootId = 'scene-root';

const isEditorOnlyHostEntity = (entity: HostSceneEntity) =>
  entity.name.toLocaleLowerCase() === 'editor camera';

const sceneRootEntity = (children: SceneEntity[]): SceneEntity => ({
  id: sceneRootId,
  name: 'Scene',
  kind: 'folder',
  active: true,
  children,
});

const resizeLimits = {
  left: { min: 220, max: 520 },
  right: { min: 300, max: 640 },
  bottom: { min: 112, max: 420 },
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const parseHostEntityId = (id: string): HostEntityId | null => {
  const [index, generation] = id.split(':').map((part) => Number.parseInt(part, 10));
  if (!Number.isInteger(index) || !Number.isInteger(generation)) {
    return null;
  }
  return { index, generation };
};

const snapshotFromMockEntity = (entity: SceneEntity | null): InspectorEntitySnapshot | null => {
  if (!entity || !entity.transform) return null;
  const hostEntity = parseHostEntityId(entity.id) ?? { index: 0, generation: 0 };
  const rotationQuaternion = eulerDegreesToQuaternion(entity.transform.rotation);
  return {
    entity: hostEntity,
    name: entity.name,
    tag: entity.kind === 'camera' ? 'Camera' : 'Untagged',
    active: entity.active,
    renderLayerMask: 1,
    transform: {
      position: entity.transform.position,
      rotationDegrees: entity.transform.rotation,
      scale: entity.transform.scale,
      rotationQuaternion,
    },
    camera: entity.kind === 'camera' ? {
      projection: 'perspective',
      fovYDegrees: 60,
      orthographicHeight: 10,
      nearPlane: 0.1,
      farPlane: 2000,
      active: true,
      clearColor: { x: 0.055, y: 0.12, z: 0.22, w: 1 },
    } : null,
    components: [
      { kind: 'transform', label: 'Transform', editable: true },
      ...(entity.kind === 'camera' ? [{ kind: 'camera', label: 'Camera', editable: true }] : []),
    ],
  };
};

export function Workbench() {
  const { layout, setLayout, resetLayout } = useWorkbenchLayout();
  const [startupState, setStartupState] = useState<StartupState | null>(null);
  const [project, setProject] = useState<ProjectSnapshot | null>(null);
  const [hostConsoleEvents, setHostConsoleEvents] = useState<ConsoleEvent[]>([]);
  const [selectedEntityId, setSelectedEntityId] = useState('camera-main');
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>('asset-scene-demo');
  const [selectedSnapshot, setSelectedSnapshot] = useState<InspectorEntitySnapshot | null>(null);
  const [selectedSnapshotLoading, setSelectedSnapshotLoading] = useState(false);
  const selectedSnapshotRevision = useRef(0);
  const [worldEnvironment, setWorldEnvironment] = useState<HostWorldEnvironment | null>(null);
  const [lastCommand, setLastCommand] = useState('Workbench ready');

  const loadAssetThumbnail = useCallback(async (path: string): Promise<string | null> => {
    if (!startupState?.engineHostConnected || !window.arc?.host) return null;
    const response = await window.arc.host.query('asset.thumbnail', { path, maxSize: 128 }) as HostResponse<HostAssetThumbnailSnapshot>;
    return response.succeeded && response.payload?.dataUrl ? response.payload.dataUrl : null;
  }, [project?.assetRoot, startupState?.engineHostConnected]);

  useEffect(() => {
    if (!activityRegistry.some((activity) => activity.id === layout.activeActivity)) {
      setLayout((current) => ({
        ...current,
        activeActivity: 'scene',
        leftVisible: true,
      }));
    }
  }, [layout.activeActivity, setLayout]);

  useEffect(() => {
    return window.arc?.host?.onLog?.((event) => {
      const entry: ConsoleEvent = {
        id: `host-log-${Date.now()}-${Math.random().toString(36).slice(2)}`,
        level: event.level,
        source: event.source,
        message: event.message,
        timestamp: event.timestamp || timestamp(),
      };
      setHostConsoleEvents((current) => [...current, entry].slice(-1000));
      setLastCommand(event.message);
    });
  }, []);

  const refreshSelectedEntity = async (entityId = selectedEntityId, connected = startupState?.engineHostConnected ?? false) => {
    const requestRevision = ++selectedSnapshotRevision.current;
    if (!connected || !window.arc?.host) {
      const mockEntity = project ? flattenScene(project.scene).find((entity) => entity.id === entityId) ?? null : null;
      if (requestRevision === selectedSnapshotRevision.current) {
        setSelectedSnapshot(snapshotFromMockEntity(mockEntity));
      }
      return;
    }
    setSelectedSnapshotLoading(true);
    try {
      const response = await window.arc.host.query('entity.selected') as HostResponse<unknown>;
      if (requestRevision !== selectedSnapshotRevision.current) return;
      if (!response.succeeded || !response.payload) {
        setSelectedSnapshot(null);
        setLastCommand(response.error || 'Could not read selected entity');
        return;
      }
      setSelectedSnapshot(parseSelectedEntitySnapshot(response.payload));
    } catch (error) {
      if (requestRevision !== selectedSnapshotRevision.current) return;
      setSelectedSnapshot(null);
      setLastCommand(error instanceof Error ? error.message : String(error));
    } finally {
      if (requestRevision === selectedSnapshotRevision.current) {
        setSelectedSnapshotLoading(false);
      }
    }
  };

  const refreshWorldEnvironment = async (entityId: string) => {
    const entity = parseHostEntityId(entityId);
    if (!entity || !window.arc?.host) {
      setWorldEnvironment(null);
      return;
    }
    try {
      const response = await window.arc.host.query('environment.state', { entity }) as HostResponse<HostWorldEnvironment>;
      setWorldEnvironment(response.succeeded && response.payload ? response.payload : null);
    } catch {
      setWorldEnvironment(null);
    }
  };

  const runCommand = async (command: CommandId) => {
    if (command === 'layout.reset') {
      resetLayout();
      setLastCommand('Layout reset');
      return;
    }

    if (command === 'file.open' || command === 'file.importScene') {
      const append = command === 'file.importScene';
      try {
        const result = await window.arc?.dialog?.openScene?.({ append });
        if (!result || result.canceled) {
          setLastCommand(append ? 'Scene import canceled' : 'Open scene canceled');
          return;
        }

        const response = result.response as HostResponse<{ entityCount?: number }> | undefined;
        if (!response?.succeeded) {
          setLastCommand(response?.error || 'Scene import failed');
          return;
        }

        await refreshProjectFromHost(result.filePath);
        const count = response.payload?.entityCount ?? 0;
        setLastCommand(`${append ? 'Imported' : 'Opened'} ${assetNameFromPath(result.filePath ?? 'scene')} (${count} entities)`);
      } catch (error) {
        setLastCommand(error instanceof Error ? error.message : String(error));
      }
      return;
    }

    const result = await executeWorkbenchCommand(command);
    setLastCommand(result.message);
  };

  const refreshProjectFromHost = async (activeScene?: string) => {
    if (!window.arc?.host) {
      return;
    }

    const [sceneResponse, assetsResponse] = await Promise.all([
      window.arc.host.query('scene.hierarchy') as Promise<HostResponse<HostSceneSnapshot>>,
      window.arc.host.query('project.assets') as Promise<HostResponse<HostProjectAssetsSnapshot>>,
    ]);

    if (!sceneResponse.succeeded || !sceneResponse.payload) {
      return;
    }

    const hostEntities = sceneResponse.payload.entities.filter((entity) => !isEditorOnlyHostEntity(entity));
    const scene = hostEntities.map((entity): SceneEntity => ({
      id: hostEntityKey(entity.entity),
      name: entity.name,
      kind: sceneKindFromHost(entity.kind),
      active: entity.active,
    }));

    const hostAssets = assetsResponse.succeeded && assetsResponse.payload ? assetsResponse.payload : null;
    const assets = hostAssets?.assets.map((asset): AssetItem => ({
      id: asset.path,
      name: assetNameFromPath(asset.path),
      path: asset.path,
      kind: assetKindFromHost(asset.kind),
      status: asset.importRunning ? 'importing' : asset.imported ? 'ready' : 'missing',
    })) ?? project?.assets ?? [];

    const selected = hostEntities.find((entity) => entity.selected) ?? hostEntities[0];
    if (selected) {
      const selectedKey = hostEntityKey(selected.entity);
      setSelectedEntityId(selectedKey);
      await refreshSelectedEntity(selectedKey, true);
    }
    const environmentEntity = hostEntities.find((entity) => entity.kind === 'environment');
    if (environmentEntity) await refreshWorldEnvironment(hostEntityKey(environmentEntity.entity));
    if (activeScene) {
      setSelectedAssetId(activeScene);
    }

    setProject((current) => ({
      ...(current ?? {
        name: hostAssets?.projectName || 'Arc Sandbox',
        root: hostAssets?.projectRoot || '',
        assetRoot: hostAssets?.assetRoot || '',
        activeScene: activeScene ?? '',
        scene: [],
        assets: [],
        console: [],
        renderStats: {
          fps: 0,
          frameTimeMs: 0,
          drawCalls: 0,
          triangles: 0,
          visibleEntities: 0,
          lights: 0,
          gpuMemoryMb: 0,
        },
      }),
      name: hostAssets?.projectName || current?.name || 'Arc Sandbox',
      root: hostAssets?.projectRoot || current?.root || '',
      assetRoot: hostAssets?.assetRoot || current?.assetRoot || '',
      activeScene: activeScene ?? current?.activeScene ?? '',
      scene,
      assets,
      console: [
        ...(current?.console ?? []),
        {
          id: `host-scene-${Date.now()}`,
          level: 'info',
          source: 'host',
          message: activeScene ? `Loaded scene asset ${assetNameFromPath(activeScene)}.` : 'Host scene snapshot refreshed.',
          timestamp: timestamp(),
        },
      ],
    }));
  };

  useEffect(() => {
    const startup = window.arc?.getStartupState?.() ?? Promise.resolve(fallbackStartupState);
    void startup
      .then(async (state) => {
        setStartupState(state);
        if (state.engineHostConnected) {
          await refreshProjectFromHost();
          return;
        }

        const snapshot = await mockHost.getProjectSnapshot();
        setProject(snapshot);
        const selected = flattenScene(snapshot.scene).find((entity) => entity.id === selectedEntityId) ?? null;
        setSelectedSnapshot(snapshotFromMockEntity(selected));
      })
      .catch(async () => {
        setStartupState(fallbackStartupState);
        setProject(await mockHost.getProjectSnapshot());
      });
  }, []);

  const selectEntity = async (entityId: string) => {
    const result = await mockHost.selectEntity(entityId);
    setSelectedEntityId(result.selectedEntityId);
    const hostEntity = parseHostEntityId(entityId);
    if (startupState?.engineHostConnected && hostEntity) {
      await window.arc.host.command('entity.select', { entity: hostEntity });
      await refreshSelectedEntity(entityId, true);
    } else {
      const mockEntity = project ? flattenScene(project.scene).find((entity) => entity.id === entityId) ?? null : null;
      setSelectedSnapshot(snapshotFromMockEntity(mockEntity));
    }
  };

  const updateWorldEnvironment = (next: HostWorldEnvironment) => {
    setWorldEnvironment(next);
    if (!startupState?.engineHostConnected) return;
    void window.arc.host.command('environment.update', { environment: next }).then((response) => {
      const result = response as HostResponse;
      setLastCommand(result.succeeded ? 'World environment updated' : result.error || 'Environment update failed');
    }).catch((error) => setLastCommand(error instanceof Error ? error.message : String(error)));
  };

  const applyWorldEnvironmentPreset = async (preset: string) => {
    if (!worldEnvironment || !startupState?.engineHostConnected) return;
    const response = await window.arc.host.command('environment.applyPreset', {
      entity: worldEnvironment.entity,
      preset,
    }) as HostResponse;
    setLastCommand(response.succeeded ? `Applied ${preset} environment preset` : response.error || 'Preset failed');
    if (response.succeeded) await refreshWorldEnvironment(hostEntityKey(worldEnvironment.entity));
  };

  const applyWorldEnvironmentHdri = async (path: string): Promise<boolean> => {
    if (!worldEnvironment) return false;
    setWorldEnvironment({ ...worldEnvironment, hdriPath: path });
    if (!startupState?.engineHostConnected) return true;
    const response = await window.arc.host.command('environment.setHdri', {
      entity: worldEnvironment.entity,
      path,
    }) as HostResponse;
    setLastCommand(response.succeeded ? 'Environment HDRI loaded' : response.error || 'HDRI load failed');
    await refreshWorldEnvironment(hostEntityKey(worldEnvironment.entity));
    return response.succeeded;
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
  const setActivityExpanded = (expanded: boolean) => setLayout((current) => ({ ...current, activityExpanded: expanded }));

  const beginPanelResize = (panel: 'left' | 'right' | 'bottom', event: ReactPointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const startX = event.clientX;
    const startY = event.clientY;
    const startLeft = layout.leftPanelWidth;
    const startRight = layout.rightPanelWidth;
    const startBottom = layout.bottomPanelHeight;

    const onPointerMove = (moveEvent: PointerEvent) => {
      setLayout((current) => {
        if (panel === 'left') {
          return {
            ...current,
            leftPanelWidth: clamp(startLeft + moveEvent.clientX - startX, resizeLimits.left.min, resizeLimits.left.max),
          };
        }

        if (panel === 'right') {
          return {
            ...current,
            rightPanelWidth: clamp(startRight - (moveEvent.clientX - startX), resizeLimits.right.min, resizeLimits.right.max),
          };
        }

        return {
          ...current,
          bottomPanelHeight: clamp(startBottom - (moveEvent.clientY - startY), resizeLimits.bottom.min, resizeLimits.bottom.max),
        };
      });
    };

    const onPointerUp = () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
  };

  const renderLeftPanel = () => {
    if (!project) {
      return <div className="side-loading">Loading workbench data...</div>;
    }

    if (layout.activeActivity === 'scene') {
      return <ExplorerPanel project={project} selectedEntityId={selectedEntityId} onSelectEntity={selectEntity} />;
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
      return <ViewportPanel project={project} startupState={startupState} onCommand={runCommand} />;
    }

    return <ViewportPanel project={project} startupState={startupState} onCommand={runCommand} />;
  };

  const renderRightPanel = (panel: WorkbenchPanelId) => {
    if (panel === 'inspector') {
      return <DataDrivenInspector
        command={async (type, payload) => {
          if (!startupState?.engineHostConnected) return { succeeded: true };
          return window.arc.host.command(type, payload) as Promise<HostResponse>;
        }}
        loading={selectedSnapshotLoading}
        snapshot={selectedSnapshot}
        onStatus={setLastCommand}
        refresh={async () => {
          if (startupState?.engineHostConnected) await refreshSelectedEntity(selectedEntityId, true);
        }}
      />;
    }
    if (panel === 'worldSettings') {
      return <WorldSettingsPanel environment={worldEnvironment} onEnvironmentChange={updateWorldEnvironment}
        assets={project?.assets ?? []} thumbnailProvider={loadAssetThumbnail}
        onEnvironmentPreset={applyWorldEnvironmentPreset} onEnvironmentHdri={applyWorldEnvironmentHdri} />;
    }
    return <LightingPanel />;
  };

  const renderBottomPanel = (panel: WorkbenchPanelId) => {
    if (panel === 'contentBrowser') {
      return <ContentBrowserPanel project={project} selectedAssetId={selectedAssetId} onSelectAsset={setSelectedAssetId} onCommand={runCommand} />;
    }

    if (panel === 'console') {
      return <ConsolePanel events={[...(project?.console ?? []), ...hostConsoleEvents]} lastCommand={lastCommand} />;
    }

    return <ContentBrowserPanel project={project} selectedAssetId={selectedAssetId} onSelectAsset={setSelectedAssetId} onCommand={runCommand} />;
  };

  const workbenchBodyStyle = {
    '--arc-left-panel-width': `${layout.leftPanelWidth}px`,
    '--arc-right-panel-width': `${layout.rightPanelWidth}px`,
    '--arc-bottom-panel-height': `${layout.bottomPanelHeight}px`,
  } as CSSProperties;

  return (
    <main className="workbench-shell">
      <MenuBar projectTitle={`${project?.name ?? 'arc editor2'}${project ? '.arcscene*' : ''}`} onCommand={runCommand} />
      <MainToolbar onCommand={runCommand} />

      <section className={[
        'workbench-body',
        'workbench-body-foundation',
        layout.activityExpanded ? 'activity-expanded' : '',
        layout.bottomVisible ? '' : 'bottom-hidden',
      ].filter(Boolean).join(' ')} style={workbenchBodyStyle}>
        <ActivityBar
          activeActivity={layout.activeActivity}
          expanded={layout.activityExpanded}
          onExpandedChange={setActivityExpanded}
          onSelectActivity={selectActivity}
        />

        {layout.leftVisible && <aside className="side-bar">{renderLeftPanel()}</aside>}
        {layout.leftVisible && (
          <ResizeHandle
            className="resize-handle-left"
            label="Resize left panel"
            onPointerDown={(event) => beginPanelResize('left', event)}
          />
        )}

        <section className="editor-region foundation-editor-region">
          <DockHost
            region="center"
            panelIds={dockPanelIds.center}
            activePanelId={layout.activeCenterPanel}
            onActivePanelChange={setActiveCenterPanel}
            renderPanel={renderCenterPanel}
          />
        </section>

        {layout.bottomVisible && (
          <>
            <ResizeHandle
              className="resize-handle-bottom"
              label="Resize bottom panel"
              onPointerDown={(event) => beginPanelResize('bottom', event)}
            />
            <DockHost
              region="bottom"
              panelIds={dockPanelIds.bottom}
              activePanelId={layout.activeBottomPanel}
              onActivePanelChange={setActiveBottomPanel}
              renderPanel={renderBottomPanel}
            />
          </>
        )}

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
        {layout.rightVisible && (
          <ResizeHandle
            className="resize-handle-right"
            label="Resize right panel"
            onPointerDown={(event) => beginPanelResize('right', event)}
          />
        )}
      </section>

      <StatusBar startupState={startupState} activeScene={project?.activeScene} lastCommand={lastCommand} />
    </main>
  );
}

function ExplorerPanel({ project, selectedEntityId, onSelectEntity }: {
  project: ProjectSnapshot;
  selectedEntityId: string;
  onSelectEntity: (entityId: string) => void;
}) {
  const [filter, setFilter] = useState('');
  const sceneTree = useMemo(() => [sceneRootEntity(project.scene)], [project.scene]);
  const filteredScene = useMemo(() => filterSceneTree(sceneTree, filter), [sceneTree, filter]);
  const allEntities = useMemo(() => flattenScene(project.scene), [project.scene]);
  const actorCount = allEntities.length;
  const selectedCount = allEntities.some((entity) => entity.id === selectedEntityId) ? 1 : 0;

  return (
    <div className="explorer-view">
      <Panel icon={<FolderTree size={14} />} title="Hierarchy">
        <label className="hierarchy-search">
          <Search size={15} />
          <input aria-label="Search hierarchy" placeholder="Search..." value={filter} onChange={(event) => setFilter(event.target.value)} />
        </label>
        <div className="hierarchy-tree">
          {filteredScene.map((entity) => (
            <SceneTreeItem key={entity.id} entity={entity} depth={0} selectedEntityId={selectedEntityId} onSelectEntity={onSelectEntity} />
          ))}
          {filteredScene.length === 0 && <div className="hierarchy-empty">No matching entities</div>}
        </div>
        <footer className="hierarchy-footer">{actorCount.toLocaleString()} actors ({selectedCount} selected)</footer>
      </Panel>
    </div>
  );
}

const normalizeFilterText = (value: string) => value.toLocaleLowerCase().replace(/[_./\\-]+/g, ' ').trim();

const fuzzyIncludes = (value: string, query: string) => {
  if (!query) {
    return true;
  }

  let index = 0;
  for (const character of value) {
    if (character === query[index]) {
      index += 1;
      if (index === query.length) {
        return true;
      }
    }
  }
  return false;
};

const entityMatchesFilter = (entity: SceneEntity, filter: string) => {
  const words = normalizeFilterText(filter).split(/\s+/).filter(Boolean);
  if (words.length === 0) {
    return true;
  }

  const haystack = normalizeFilterText(`${entity.name} ${entity.kind} ${(entity.components ?? []).join(' ')}`);
  return words.every((word) => haystack.includes(word) || fuzzyIncludes(haystack, word));
};

const filterSceneTree = (entities: SceneEntity[], filter: string): SceneEntity[] => {
  const normalized = normalizeFilterText(filter);
  if (!normalized) {
    return entities;
  }

  return entities.flatMap((entity) => {
    const children = filterSceneTree(entity.children ?? [], normalized);
    if (entityMatchesFilter(entity, normalized) || children.length > 0) {
      return [{ ...entity, children }];
    }
    return [];
  });
};

function AssetExplorerPanel({ project, selectedAssetId, onSelectAsset }: {
  project: ProjectSnapshot;
  selectedAssetId: string | null;
  onSelectAsset: (assetId: string) => void;
}) {
  return (
    <Panel icon={<Database size={14} />} title="Assets">
      <TreeSection title="Project Assets">
        {project.assets.map((asset) => <AssetRow key={asset.id} asset={asset} selected={asset.id === selectedAssetId} onSelect={() => onSelectAsset(asset.id)} />)}
      </TreeSection>
    </Panel>
  );
}

function WorldSettingsPanel({ environment, assets, thumbnailProvider, onEnvironmentChange, onEnvironmentPreset, onEnvironmentHdri }: {
  environment: HostWorldEnvironment | null;
  assets: ReadonlyArray<AssetItem>;
  thumbnailProvider: (path: string) => Promise<string | null>;
  onEnvironmentChange: (environment: HostWorldEnvironment) => void;
  onEnvironmentPreset: (preset: string) => void;
  onEnvironmentHdri: (path: string) => Promise<boolean> | boolean | void;
}) {
  return (
    <section className="world-settings-panel">
      {environment
        ? <WorldEnvironmentInspector environment={environment} assets={assets} thumbnailProvider={thumbnailProvider}
          onChange={onEnvironmentChange} onPreset={onEnvironmentPreset} onHdri={onEnvironmentHdri} />
        : <PlaceholderPanel icon={<Settings />} title="World Settings" text="No world environment is available in this scene." />}
    </section>
  );
}

function LightingPanel() {
  return (
    <section className="lighting-panel-placeholder">
      <Lightbulb size={25} />
      <h3>Lighting</h3>
      <p>Lighting component editing will use the same schema-driven controls in a later milestone.</p>
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
        <UiButton onClick={() => onCommand('assets.import')} variant="toolbar">+ Add</UiButton>
        <UiButton onClick={() => onCommand('assets.import')} variant="toolbar">Import</UiButton>
        <UiButton onClick={() => onCommand('assets.saveAll')} variant="toolbar">Save All</UiButton>
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

function SettingsPanel({ onResetLayout }: { onResetLayout: () => void }) {
  return <Panel icon={<Settings size={14} />} title="Settings"><UiButton className="settings-action" onClick={onResetLayout} variant="toolbar">Reset Layout</UiButton><PropertyRow label="Theme" value="Arc Dark" /><PropertyRow label="Host" value="Mock" /></Panel>;
}

function Panel({ children, icon, title }: { children: ReactNode; icon: ReactNode; title: string }) {
  return (
    <UiPanel className="workbench-panel">
      <UiTabs className="dock-tabs panel-tabs">
        <div className="dock-tab-strip">
          <UiTab active className="dock-tab panel-tab" title={title}>
            {icon}
            <span>{title}</span>
            <X className="dock-tab-close" size={12} />
          </UiTab>
        </div>
        <UiIconButton className="dock-header-action" label={`${title} panel actions`}>
          <MoreVertical size={14} />
        </UiIconButton>
      </UiTabs>
      <div>{children}</div>
    </UiPanel>
  );
}

function ResizeHandle({ className, label, onPointerDown }: {
  className: string;
  label: string;
  onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>) => void;
}) {
  return <button aria-label={label} className={`resize-handle ${className}`} onPointerDown={onPointerDown} type="button" />;
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
  const selectable = entity.id !== sceneRootId;
  return (
    <div>
      <UiTreeRow
        className={`tree-row entity-row entity-${entity.kind}`}
        depth={depth}
        selected={selectable && entity.id === selectedEntityId}
        meta={!entity.active && <small>off</small>}
        onClick={() => selectable && onSelectEntity(entity.id)}
      >
        {hasChildren ? <ChevronDown size={13} /> : <ChevronRight size={13} className="ghost" />}
        <EntityIcon kind={entity.kind} />
        <span>{entity.name}</span>
      </UiTreeRow>
      {entity.children?.map((child) => <SceneTreeItem key={child.id} entity={child} depth={depth + 1} selectedEntityId={selectedEntityId} onSelectEntity={onSelectEntity} />)}
    </div>
  );
}

function EntityIcon({ kind }: { kind: SceneEntity['kind'] }) {
  if (kind === 'camera') return <Code2 className="entity-icon entity-icon-camera" size={14} />;
  if (kind === 'light') return <Lightbulb className="entity-icon entity-icon-light" size={14} />;
  if (kind === 'folder') return <Folder className="entity-icon entity-icon-folder" size={14} />;
  return <Box className="entity-icon entity-icon-mesh" size={14} />;
}

function AssetRow({ asset, selected, onSelect }: { asset: AssetItem; selected: boolean; onSelect: () => void }) {
  return <UiTreeRow className="tree-row" selected={selected} meta={<small>{asset.status}</small>} onClick={onSelect}><AssetIcon kind={asset.kind} /><span>{asset.name}</span></UiTreeRow>;
}

function AssetCard({ asset, selected, onSelect }: { asset: AssetItem; selected: boolean; onSelect: () => void }) {
  return <UiButton className={selected ? 'asset-card-foundation selected' : 'asset-card-foundation'} draggable={asset.kind === 'texture'} onDragStart={(event) => {
    if (asset.kind !== 'texture') return;
    event.dataTransfer.setData('application/x-arc-asset', asset.path);
    event.dataTransfer.setData('application/x-arc-environment', asset.path);
  }} onClick={onSelect} variant="default"><AssetIcon kind={asset.kind} /><strong>{asset.name}</strong><span>{asset.kind}</span></UiButton>;
}

function AssetIcon({ kind }: { kind: AssetItem['kind'] }) {
  if (kind === 'scene') return <FileText size={14} />;
  if (kind === 'shader') return <FileCode2 size={14} />;
  if (kind === 'folder') return <Folder size={14} />;
  return <Database size={14} />;
}

function PropertyRow({ label, value }: { label: string; value: ReactNode }) {
  return <div className="ui-property-row property-row"><span>{label}</span><strong>{value}</strong></div>;
}

function LogLine({ level, text }: { level: ConsoleEvent['level'] | 'warning' | 'debug'; text: string }) {
  return <code className={`log-line ${level}`}>{level === 'warning' && <AlertTriangle size={13} />} {text}</code>;
}

function PlaceholderPanel({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return <div className="empty-state">{icon}<h3>{title}</h3><p>{text}</p></div>;
}

void defaultWorkbenchLayout;
