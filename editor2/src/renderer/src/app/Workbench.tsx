import { useEffect, useMemo, useState } from 'react';
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
  Link,
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
import type { AssetItem, ConsoleEvent, ProjectSnapshot, SceneEntity, Transform, Vec3 } from '../services/mockHost';
import { UiButton, UiIconButton, UiPanel, UiTab, UiTabs, UiTextInput, UiTreeRow } from '../ui';
import { ViewportPanel } from '../viewport/ViewportPanel';

import './workbench.css';

type HostResponse<TPayload = unknown> = {
  succeeded: boolean;
  error?: string;
  payload?: TPayload;
};

type HostEntityId = {
  index: number;
  generation: number;
};

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
  assets: HostAssetSnapshot[];
};

type HostCloudLayer = {
  enabled: boolean;
  coverage: number;
  density: number;
  altitude: number;
  thickness: number;
  scale: number;
  detail: number;
  softness: number;
  windX: number;
  windY: number;
  windSpeed: number;
  lightingStrength: number;
  silverLining: number;
};

type HostWorldEnvironment = {
  entity: HostEntityId;
  enabled: boolean;
  skyVisible: boolean;
  affectLighting: boolean;
  skySource: 'physicalAtmosphere' | 'hdri' | 'solidColor';
  solidColor: Vec3;
  hdriPath: string;
  hdriRotationDegrees: number;
  radianceIntensity: number;
  planetRadius: number;
  atmosphereRadius: number;
  rayleighStrength: number;
  mieStrength: number;
  ozoneStrength: number;
  atmosphereTint: Vec3;
  groundAlbedo: Vec3;
  mieAnisotropy: number;
  rayleighScaleHeight: number;
  mieScaleHeight: number;
  multiScatteringFactor: number;
  exposure: number;
  sunDiskSize: number;
  sunDiskIntensity: number;
  sunMode: 'manualLight' | 'geographic';
  timeMode: 'fixed' | 'simulated' | 'systemClock';
  latitudeDegrees: number;
  longitudeDegrees: number;
  northOffsetDegrees: number;
  year: number;
  month: number;
  day: number;
  localTimeHours: number;
  utcOffsetHours: number;
  playing: boolean;
  loopDay: boolean;
  timeScale: number;
  automaticSunLight: boolean;
  sunIntensityMultiplier: number;
  sunTemperatureMultiplier: number;
  moonEnabled: boolean;
  automaticMoonPhase: boolean;
  moonPhase: number;
  moonIntensity: number;
  moonAngularRadiusDegrees: number;
  starsEnabled: boolean;
  starDensity: number;
  starIntensity: number;
  starTwinkle: number;
  cloudsEnabled: boolean;
  cloudShadows: boolean;
  cumulus: HostCloudLayer;
  cirrus: HostCloudLayer;
  fogEnabled: boolean;
  fogColor: Vec3;
  fogDensity: number;
  fogHeightFalloff: number;
  fogStartDistance: number;
  fogMaxOpacity: number;
  fogSunScattering: number;
  lightingEnabled: boolean;
  lightingSource: 'followSky' | 'hdri' | 'constantColor';
  lightingColor: Vec3;
  diffuseIntensity: number;
  specularIntensity: number;
};

const fallbackStartupState: StartupState = {
  appVersion: 'dev',
  engineHostConnected: false,
  viewportMode: 'placeholder',
};

const defaultTransform = {
  position: { x: 0, y: 0, z: 0 },
  rotation: { x: 0, y: 0, z: 0 },
  scale: { x: 1, y: 1, z: 1 },
};

const hostEntityKey = (entity: HostEntityId) => `${entity.index}:${entity.generation}`;

const sceneKindFromHost = (kind: HostSceneEntity['kind']): SceneEntity['kind'] => {
  if (kind === 'camera' || kind === 'light' || kind === 'environment' || kind === 'mesh') {
    return kind;
  }
  return 'mesh';
};

const componentsForHostEntity = (kind: HostSceneEntity['kind']): string[] => {
  if (kind === 'camera') return ['Transform', 'Camera'];
  if (kind === 'light') return ['Transform', 'Light'];
  if (kind === 'environment') return ['Environment'];
  return ['Transform', 'Mesh Renderer'];
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
  components: [],
  transform: defaultTransform,
});

const resizeLimits = {
  left: { min: 220, max: 520 },
  right: { min: 260, max: 560 },
  bottom: { min: 112, max: 420 },
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

type InspectorVectorField = {
  id: keyof SceneEntity['transform'];
  label: string;
  mode: 'position' | 'rotationDegrees' | 'scale';
  lockable?: boolean;
};

type VectorAxis = keyof Vec3;

type InspectorComponentSchema = {
  id: string;
  title: string;
  fields: InspectorVectorField[];
};

const inspectorComponentSchemas: InspectorComponentSchema[] = [
  {
    id: 'transform',
    title: 'Transform',
    fields: [
      { id: 'position', label: 'Location', mode: 'position' },
      { id: 'rotation', label: 'Rotation', mode: 'rotationDegrees' },
      { id: 'scale', label: 'Scale', mode: 'scale', lockable: true },
    ],
  },
];

const updateEntityTransform = (
  entities: SceneEntity[],
  entityId: string,
  transform: Transform,
): SceneEntity[] => entities.map((entity) => ({
  ...entity,
  transform: entity.id === entityId ? transform : entity.transform,
  children: entity.children ? updateEntityTransform(entity.children, entityId, transform) : entity.children,
}));

const parseHostEntityId = (id: string): HostEntityId | null => {
  const [index, generation] = id.split(':').map((part) => Number.parseInt(part, 10));
  if (!Number.isInteger(index) || !Number.isInteger(generation)) {
    return null;
  }
  return { index, generation };
};

const degreesToRadians = (degrees: number) => degrees * Math.PI / 180;

const eulerDegreesToQuaternion = (rotation: Vec3) => {
  const halfX = degreesToRadians(rotation.x) * 0.5;
  const halfY = degreesToRadians(rotation.y) * 0.5;
  const halfZ = degreesToRadians(rotation.z) * 0.5;
  const cx = Math.cos(halfX);
  const sx = Math.sin(halfX);
  const cy = Math.cos(halfY);
  const sy = Math.sin(halfY);
  const cz = Math.cos(halfZ);
  const sz = Math.sin(halfZ);

  return {
    x: sx * cy * cz + cx * sy * sz,
    y: cx * sy * cz - sx * cy * sz,
    z: cx * cy * sz + sx * sy * cz,
    w: cx * cy * cz - sx * sy * sz,
  };
};

const transformToHostPayload = (transform: Transform) => ({
  position: transform.position,
  rotation: eulerDegreesToQuaternion(transform.rotation),
  scale: transform.scale,
});

export function Workbench() {
  const { layout, setLayout, resetLayout } = useWorkbenchLayout();
  const [startupState, setStartupState] = useState<StartupState | null>(null);
  const [project, setProject] = useState<ProjectSnapshot | null>(null);
  const [hostConsoleEvents, setHostConsoleEvents] = useState<ConsoleEvent[]>([]);
  const [selectedEntityId, setSelectedEntityId] = useState('camera-main');
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>('asset-scene-demo');
  const [worldEnvironment, setWorldEnvironment] = useState<HostWorldEnvironment | null>(null);
  const [lastCommand, setLastCommand] = useState('Workbench ready');

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
      components: componentsForHostEntity(entity.kind),
      transform: defaultTransform,
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
      setSelectedEntityId(hostEntityKey(selected.entity));
    }
    if (activeScene) {
      setSelectedAssetId(activeScene);
    }

    setProject((current) => ({
      ...(current ?? {
        name: hostAssets?.projectName || 'Arc Sandbox',
        root: hostAssets?.projectRoot || '',
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

        setProject(await mockHost.getProjectSnapshot());
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
      await refreshWorldEnvironment(entityId);
    } else {
      setWorldEnvironment(null);
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

  const applyWorldEnvironmentHdri = async (path: string) => {
    if (!worldEnvironment || !startupState?.engineHostConnected) return;
    const response = await window.arc.host.command('environment.setHdri', {
      entity: worldEnvironment.entity,
      path,
    }) as HostResponse;
    setLastCommand(response.succeeded ? 'Environment HDRI loaded' : response.error || 'HDRI load failed');
    if (response.succeeded) await refreshWorldEnvironment(hostEntityKey(worldEnvironment.entity));
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

  const updateSelectedTransformField = (field: keyof Transform, axis: VectorAxis, value: number) => {
    if (!selectedEntity) {
      return;
    }

    const nextTransform: Transform = {
      ...selectedEntity.transform,
      [field]: {
        ...selectedEntity.transform[field],
        [axis]: value,
      },
    };

    setProject((current) => current ? {
      ...current,
      scene: updateEntityTransform(current.scene, selectedEntity.id, nextTransform),
    } : current);

    const hostEntity = parseHostEntityId(selectedEntity.id);
    if (startupState?.engineHostConnected && hostEntity) {
      void window.arc.host.command('entity.setTransform', {
        entity: hostEntity,
        transform: transformToHostPayload(nextTransform),
      }).catch((error) => {
        setLastCommand(error instanceof Error ? error.message : String(error));
      });
    }
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
      return <InspectorPanel entity={selectedEntity} asset={selectedAsset} environment={worldEnvironment} onEnvironmentChange={updateWorldEnvironment} onEnvironmentPreset={applyWorldEnvironmentPreset} onEnvironmentHdri={applyWorldEnvironmentHdri} onTransformFieldChange={updateSelectedTransformField} />;
    }

    return <InspectorPanel entity={selectedEntity} asset={selectedAsset} environment={worldEnvironment} onEnvironmentChange={updateWorldEnvironment} onEnvironmentPreset={applyWorldEnvironmentPreset} onEnvironmentHdri={applyWorldEnvironmentHdri} onTransformFieldChange={updateSelectedTransformField} />;
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

  const haystack = normalizeFilterText(`${entity.name} ${entity.kind} ${entity.components.join(' ')}`);
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

function InspectorPanel({ entity, asset, environment, onEnvironmentChange, onEnvironmentPreset, onEnvironmentHdri, onTransformFieldChange }: {
  entity: SceneEntity | null;
  asset: AssetItem | null;
  environment: HostWorldEnvironment | null;
  onEnvironmentChange: (environment: HostWorldEnvironment) => void;
  onEnvironmentPreset: (preset: string) => void;
  onEnvironmentHdri: (path: string) => void;
  onTransformFieldChange: (field: keyof Transform, axis: VectorAxis, value: number) => void;
}) {
  return (
    <section className="inspector foundation-inspector">
      {entity ? (
        <>
          <InspectorEntityHeader entity={entity} />
          <label className="inspector-search">
            <Search size={15} />
            <input aria-label="Search components" placeholder="Search components..." />
          </label>
          <div className="inspector-components">
            {environment ? (
              <WorldEnvironmentInspector environment={environment} onChange={onEnvironmentChange} onPreset={onEnvironmentPreset} onHdri={onEnvironmentHdri} />
            ) : inspectorComponentSchemas.map((component) => (
              <InspectorComponent
                key={component.id}
                component={component}
                entity={entity}
                onTransformFieldChange={onTransformFieldChange}
              />
            ))}
          </div>
          <section className="component-list">
            <h3>Components</h3>
            {entity.components.length ? entity.components.map((component) => <UiButton key={component} variant="toolbar">{component}</UiButton>) : <span>No components</span>}
          </section>
        </>
      ) : <PlaceholderPanel icon={<Settings />} title="Nothing selected" text="Select an entity or asset." />}
      {asset && <section className="asset-inspector"><h3>Selected Asset</h3><PropertyRow label="Name" value={asset.name} /><PropertyRow label="Path" value={asset.path} /><PropertyRow label="Status" value={asset.status} /></section>}
    </section>
  );
}

function InspectorEntityHeader({ entity }: { entity: SceneEntity }) {
  return (
    <header className="inspector-entity-header">
      <label className="inspector-active-toggle">
        <input aria-label={`${entity.name} active`} checked={entity.active} readOnly type="checkbox" />
      </label>
      <h2>{entity.name}</h2>
      <label className="inspector-static-toggle">
        <input aria-label={`${entity.name} static`} readOnly type="checkbox" />
        <span>Static</span>
      </label>
      <UiIconButton className="inspector-header-action" label="Entity actions">
        <MoreVertical size={14} />
      </UiIconButton>
      <PropertyRow label="Kind" value={entity.kind} />
      <PropertyRow label="Active" value={entity.active ? 'true' : 'false'} />
    </header>
  );
}

function InspectorComponent({ component, entity, onTransformFieldChange }: {
  component: InspectorComponentSchema;
  entity: SceneEntity;
  onTransformFieldChange: (field: keyof Transform, axis: VectorAxis, value: number) => void;
}) {
  return (
    <section className="inspector-component">
      <header className="inspector-component-header">
        <ChevronDown size={14} />
        <span>{component.title}</span>
        <ChevronDown className="inspector-component-menu" size={14} />
      </header>
      <div className="inspector-component-body">
        {component.fields.map((field) => (
          <InspectorVectorField
            key={field.id}
            field={field}
            onAxisChange={(axis, value) => onTransformFieldChange(field.id, axis, value)}
            value={entity.transform[field.id]}
          />
        ))}
      </div>
    </section>
  );
}

function WorldEnvironmentInspector({ environment, onChange, onPreset, onHdri }: {
  environment: HostWorldEnvironment;
  onChange: (environment: HostWorldEnvironment) => void;
  onPreset: (preset: string) => void;
  onHdri: (path: string) => void;
}) {
  const [hdriPath, setHdriPath] = useState(environment.hdriPath);
  useEffect(() => setHdriPath(environment.hdriPath), [environment.hdriPath]);
  const patch = <K extends keyof HostWorldEnvironment>(key: K, value: HostWorldEnvironment[K]) =>
    onChange({ ...environment, [key]: value });

  return (
    <div className="environment-inspector">
      <section className="environment-presets">
        {[
          ['clearDay', 'Clear Day'],
          ['alpineLateMorning', 'Alpine'],
          ['goldenHour', 'Golden Hour'],
          ['overcast', 'Overcast'],
          ['night', 'Night'],
          ['indoorNeutral', 'Indoor'],
        ].map(([id, label]) => <UiButton key={id} onClick={() => onPreset(id)} variant="toolbar">{label}</UiButton>)}
      </section>

      <EnvironmentSection title="General">
        <EnvironmentToggle label="Enabled" value={environment.enabled} onChange={(value) => patch('enabled', value)} />
        <EnvironmentToggle label="Sky Visible" value={environment.skyVisible} onChange={(value) => patch('skyVisible', value)} />
        <EnvironmentToggle label="Affect Lighting" value={environment.affectLighting} onChange={(value) => patch('affectLighting', value)} />
      </EnvironmentSection>

      <EnvironmentSection title="Sky Source">
        <EnvironmentSelect label="Source" value={environment.skySource} options={[
          ['physicalAtmosphere', 'Physical Atmosphere'], ['hdri', 'HDRI'], ['solidColor', 'Solid Color'],
        ]} onChange={(value) => patch('skySource', value as HostWorldEnvironment['skySource'])} />
        <EnvironmentNumber label="Radiance" min={0} step={0.05} value={environment.radianceIntensity} onChange={(value) => patch('radianceIntensity', value)} />
        <EnvironmentNumber label="Rotation" step={1} value={environment.hdriRotationDegrees} onChange={(value) => patch('hdriRotationDegrees', value)} />
        <label className="environment-path"><span>HDRI</span><input onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); const path = event.dataTransfer.getData('application/x-arc-environment'); if (path) { setHdriPath(path); onHdri(path); } }} onChange={(event) => setHdriPath(event.target.value)} placeholder="environments/studio.hdr" value={hdriPath} /></label>
        <UiButton onClick={() => onHdri(hdriPath)} variant="toolbar">Load HDRI</UiButton>
      </EnvironmentSection>

      <EnvironmentSection title="Sun & Time">
        <EnvironmentSelect label="Sun Position" value={environment.sunMode} options={[
          ['manualLight', 'Manual Light'], ['geographic', 'Geographic'],
        ]} onChange={(value) => patch('sunMode', value as HostWorldEnvironment['sunMode'])} />
        <EnvironmentSelect label="Clock" value={environment.timeMode} options={[
          ['fixed', 'Fixed'], ['simulated', 'Simulated'], ['systemClock', 'System Clock'],
        ]} onChange={(value) => patch('timeMode', value as HostWorldEnvironment['timeMode'])} />
        <EnvironmentToggle label="Play" value={environment.playing} onChange={(value) => patch('playing', value)} />
        <EnvironmentNumber label="Time of Day" min={0} max={23.999} step={0.05} value={environment.localTimeHours} onChange={(value) => patch('localTimeHours', value)} />
        <EnvironmentNumber label="Time Scale" min={0} step={1} value={environment.timeScale} onChange={(value) => patch('timeScale', value)} />
        <EnvironmentNumber label="Latitude" min={-90} max={90} step={0.1} value={environment.latitudeDegrees} onChange={(value) => patch('latitudeDegrees', value)} />
        <EnvironmentNumber label="Longitude" min={-180} max={180} step={0.1} value={environment.longitudeDegrees} onChange={(value) => patch('longitudeDegrees', value)} />
        <EnvironmentNumber label="UTC Offset" min={-14} max={14} step={0.5} value={environment.utcOffsetHours} onChange={(value) => patch('utcOffsetHours', value)} />
        <EnvironmentNumber label="North Offset" step={1} value={environment.northOffsetDegrees} onChange={(value) => patch('northOffsetDegrees', value)} />
        <EnvironmentToggle label="Loop Day" value={environment.loopDay} onChange={(value) => patch('loopDay', value)} />
        <EnvironmentToggle label="Automatic Sun" value={environment.automaticSunLight} onChange={(value) => patch('automaticSunLight', value)} />
        <EnvironmentNumber label="Sun Intensity" min={0} step={0.05} value={environment.sunIntensityMultiplier} onChange={(value) => patch('sunIntensityMultiplier', value)} />
        <EnvironmentNumber label="Sun Temperature" min={0.1} step={0.05} value={environment.sunTemperatureMultiplier} onChange={(value) => patch('sunTemperatureMultiplier', value)} />
        <div className="environment-date">
          <EnvironmentNumber label="Year" min={1} max={9999} step={1} value={environment.year} onChange={(value) => patch('year', Math.trunc(value))} />
          <EnvironmentNumber label="Month" min={1} max={12} step={1} value={environment.month} onChange={(value) => patch('month', Math.trunc(value))} />
          <EnvironmentNumber label="Day" min={1} max={31} step={1} value={environment.day} onChange={(value) => patch('day', Math.trunc(value))} />
        </div>
      </EnvironmentSection>

      <EnvironmentSection title="Atmosphere" advanced>
        <EnvironmentNumber label="Rayleigh" min={0} step={0.02} value={environment.rayleighStrength} onChange={(value) => patch('rayleighStrength', value)} />
        <EnvironmentNumber label="Mie / Haze" min={0} step={0.02} value={environment.mieStrength} onChange={(value) => patch('mieStrength', value)} />
        <EnvironmentNumber label="Ozone" min={0} step={0.02} value={environment.ozoneStrength} onChange={(value) => patch('ozoneStrength', value)} />
        <EnvironmentNumber label="Mie Anisotropy" min={-0.98} max={0.98} step={0.01} value={environment.mieAnisotropy} onChange={(value) => patch('mieAnisotropy', value)} />
        <EnvironmentNumber label="Exposure" min={0} step={0.05} value={environment.exposure} onChange={(value) => patch('exposure', value)} />
        <EnvironmentNumber label="Rayleigh Height km" min={0.01} step={0.1} value={environment.rayleighScaleHeight} onChange={(value) => patch('rayleighScaleHeight', value)} />
        <EnvironmentNumber label="Mie Height km" min={0.01} step={0.1} value={environment.mieScaleHeight} onChange={(value) => patch('mieScaleHeight', value)} />
        <EnvironmentNumber label="Multi Scattering" min={0} step={0.05} value={environment.multiScatteringFactor} onChange={(value) => patch('multiScatteringFactor', value)} />
        <EnvironmentNumber label="Sun Disk Size" min={0} step={0.001} value={environment.sunDiskSize} onChange={(value) => patch('sunDiskSize', value)} />
        <EnvironmentNumber label="Sun Disk Power" min={0} step={0.05} value={environment.sunDiskIntensity} onChange={(value) => patch('sunDiskIntensity', value)} />
        <EnvironmentNumber label="Planet Radius km" min={1} step={1} value={environment.planetRadius} onChange={(value) => patch('planetRadius', value)} />
        <EnvironmentNumber label="Atmosphere Radius km" min={1} step={1} value={environment.atmosphereRadius} onChange={(value) => patch('atmosphereRadius', value)} />
      </EnvironmentSection>

      <EnvironmentSection title="Night Sky">
        <EnvironmentToggle label="Moon" value={environment.moonEnabled} onChange={(value) => patch('moonEnabled', value)} />
        <EnvironmentToggle label="Automatic Phase" value={environment.automaticMoonPhase} onChange={(value) => patch('automaticMoonPhase', value)} />
        <EnvironmentNumber label="Moon Phase" min={0} max={1} step={0.01} value={environment.moonPhase} onChange={(value) => patch('moonPhase', value)} />
        <EnvironmentNumber label="Moon Brightness" min={0} step={0.02} value={environment.moonIntensity} onChange={(value) => patch('moonIntensity', value)} />
        <EnvironmentNumber label="Moon Angular Radius" min={0.01} step={0.01} value={environment.moonAngularRadiusDegrees} onChange={(value) => patch('moonAngularRadiusDegrees', value)} />
        <EnvironmentToggle label="Stars" value={environment.starsEnabled} onChange={(value) => patch('starsEnabled', value)} />
        <EnvironmentNumber label="Star Density" min={0} max={1} step={0.01} value={environment.starDensity} onChange={(value) => patch('starDensity', value)} />
        <EnvironmentNumber label="Star Intensity" min={0} step={0.05} value={environment.starIntensity} onChange={(value) => patch('starIntensity', value)} />
        <EnvironmentNumber label="Star Twinkle" min={0} max={1} step={0.01} value={environment.starTwinkle} onChange={(value) => patch('starTwinkle', value)} />
      </EnvironmentSection>

      <EnvironmentSection title="Clouds">
        <EnvironmentToggle label="Enabled" value={environment.cloudsEnabled} onChange={(value) => patch('cloudsEnabled', value)} />
        <EnvironmentToggle label="Cloud Shadows" value={environment.cloudShadows} onChange={(value) => patch('cloudShadows', value)} />
        <CloudLayerEditor label="Cumulus" layer={environment.cumulus} onChange={(value) => patch('cumulus', value)} />
        <CloudLayerEditor label="Cirrus" layer={environment.cirrus} onChange={(value) => patch('cirrus', value)} />
      </EnvironmentSection>

      <EnvironmentSection title="Fog">
        <EnvironmentToggle label="Enabled" value={environment.fogEnabled} onChange={(value) => patch('fogEnabled', value)} />
        <EnvironmentNumber label="Density" min={0} step={0.001} value={environment.fogDensity} onChange={(value) => patch('fogDensity', value)} />
        <EnvironmentNumber label="Height Falloff" min={0} step={0.01} value={environment.fogHeightFalloff} onChange={(value) => patch('fogHeightFalloff', value)} />
        <EnvironmentNumber label="Start Distance" min={0} step={1} value={environment.fogStartDistance} onChange={(value) => patch('fogStartDistance', value)} />
        <EnvironmentNumber label="Max Opacity" min={0} max={1} step={0.01} value={environment.fogMaxOpacity} onChange={(value) => patch('fogMaxOpacity', value)} />
      </EnvironmentSection>

      <EnvironmentSection title="Environment Lighting">
        <EnvironmentToggle label="Enabled" value={environment.lightingEnabled} onChange={(value) => patch('lightingEnabled', value)} />
        <EnvironmentSelect label="Source" value={environment.lightingSource} options={[
          ['followSky', 'Follow Sky'], ['hdri', 'HDRI'], ['constantColor', 'Constant Color'],
        ]} onChange={(value) => patch('lightingSource', value as HostWorldEnvironment['lightingSource'])} />
        <EnvironmentNumber label="Diffuse" min={0} step={0.05} value={environment.diffuseIntensity} onChange={(value) => patch('diffuseIntensity', value)} />
        <EnvironmentNumber label="Specular" min={0} step={0.05} value={environment.specularIntensity} onChange={(value) => patch('specularIntensity', value)} />
      </EnvironmentSection>
    </div>
  );
}

function EnvironmentSection({ title, children, advanced = false }: { title: string; children: ReactNode; advanced?: boolean }) {
  return <details className="environment-section" open={!advanced}><summary><ChevronDown size={14} />{title}{advanced && <small>Advanced</small>}</summary><div>{children}</div></details>;
}

function EnvironmentToggle({ label, value, onChange }: { label: string; value: boolean; onChange: (value: boolean) => void }) {
  return <label className="environment-control"><span>{label}</span><input checked={value} onChange={(event) => onChange(event.target.checked)} type="checkbox" /></label>;
}

function EnvironmentNumber({ label, value, onChange, min, max, step }: { label: string; value: number; onChange: (value: number) => void; min?: number; max?: number; step?: number }) {
  return <label className="environment-control"><span>{label}</span><input min={min} max={max} step={step} type="number" value={Number.isFinite(value) ? value : 0} onChange={(event) => { const next = Number(event.target.value); if (Number.isFinite(next)) onChange(next); }} /></label>;
}

function EnvironmentSelect({ label, value, options, onChange }: { label: string; value: string; options: string[][]; onChange: (value: string) => void }) {
  return <label className="environment-control"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}>{options.map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>;
}

function CloudLayerEditor({ label, layer, onChange }: { label: string; layer: HostCloudLayer; onChange: (layer: HostCloudLayer) => void }) {
  const patch = <K extends keyof HostCloudLayer>(key: K, value: HostCloudLayer[K]) => onChange({ ...layer, [key]: value });
  return <fieldset className="cloud-layer"><legend>{label}</legend><EnvironmentToggle label="Enabled" value={layer.enabled} onChange={(value) => patch('enabled', value)} /><EnvironmentNumber label="Coverage" min={0} max={1} step={0.01} value={layer.coverage} onChange={(value) => patch('coverage', value)} /><EnvironmentNumber label="Density" min={0} max={1} step={0.01} value={layer.density} onChange={(value) => patch('density', value)} /><EnvironmentNumber label="Altitude" min={0} step={50} value={layer.altitude} onChange={(value) => patch('altitude', value)} /><EnvironmentNumber label="Thickness" min={0} step={10} value={layer.thickness} onChange={(value) => patch('thickness', value)} /><EnvironmentNumber label="Wind Speed" min={0} step={0.5} value={layer.windSpeed} onChange={(value) => patch('windSpeed', value)} /><EnvironmentNumber label="Silver Lining" min={0} max={1} step={0.01} value={layer.silverLining} onChange={(value) => patch('silverLining', value)} /></fieldset>;
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
  return <UiButton className={selected ? 'asset-card-foundation selected' : 'asset-card-foundation'} draggable={asset.kind === 'texture'} onDragStart={(event) => { if (asset.kind === 'texture') event.dataTransfer.setData('application/x-arc-environment', asset.path); }} onClick={onSelect} variant="default"><AssetIcon kind={asset.kind} /><strong>{asset.name}</strong><span>{asset.kind}</span></UiButton>;
}

function AssetIcon({ kind }: { kind: AssetItem['kind'] }) {
  if (kind === 'scene') return <FileText size={14} />;
  if (kind === 'shader') return <FileCode2 size={14} />;
  if (kind === 'folder') return <Folder size={14} />;
  return <Database size={14} />;
}

function InspectorVectorField({ field, onAxisChange, value }: {
  field: InspectorVectorField;
  onAxisChange: (axis: VectorAxis, value: number) => void;
  value: Vec3;
}) {
  return (
    <div className={`inspector-vector-field inspector-vector-${field.id}`}>
      <div className="inspector-vector-label">
        <span>{field.label}</span>
        {field.lockable && (
          <UiIconButton className="inspector-vector-lock" label={`Lock ${field.label.toLocaleLowerCase()} axes`}>
            <Link size={14} />
          </UiIconButton>
        )}
      </div>
      <div className="inspector-axis-group">
        <AxisField axis="x" mode={field.mode} onChange={onAxisChange} value={value.x} />
        <AxisField axis="y" mode={field.mode} onChange={onAxisChange} value={value.y} />
        <AxisField axis="z" mode={field.mode} onChange={onAxisChange} value={value.z} />
      </div>
    </div>
  );
}

function AxisField({ axis, mode, onChange, value }: {
  axis: VectorAxis;
  mode: InspectorVectorField['mode'];
  onChange: (axis: VectorAxis, value: number) => void;
  value: number;
}) {
  const formattedValue = mode === 'rotationDegrees' ? `${formatNumber(value, 1)}°` : formatNumber(value, mode === 'scale' ? 1 : 2);
  return (
    <label className={`inspector-axis-field axis-${axis}`}>
      <span>{axis.toUpperCase()}</span>
      <input
        onChange={(event) => {
          const parsed = Number.parseFloat(event.target.value.replace('°', ''));
          if (Number.isFinite(parsed)) {
            onChange(axis, parsed);
          }
        }}
        value={formattedValue}
      />
    </label>
  );
}

function formatNumber(value: number, digits: number) {
  return Number(value).toFixed(digits);
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
