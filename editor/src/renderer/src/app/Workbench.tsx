import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import {
  Box,
  ChevronDown,
  ChevronRight,
  Code2,
  Database,
  FileCode2,
  FileText,
  Folder,
  FolderTree,
  Lightbulb,
  Mountain,
  Lock,
  Unlock,
  Copy,
  Eye,
  EyeOff,
  MoreVertical,
  Plus,
  Search,
  Settings,
  Trash2,
  X,
} from 'lucide-react';

import { activityRegistry, dockPanelIds, getPanel } from './panelRegistry';
import { CommandPalette } from './CommandPalette';
import { commandRegistry } from './commandRegistry';
import { KeybindingService } from './keybindingService';
import { defaultWorkbenchLayout, useWorkbenchLayout } from './workbenchStore';
import type { ActivityId, CommandContext, CommandId, StartupState, WorkbenchPanelId } from './workbenchTypes';
import { ActivityBar } from '../layout/ActivityBar';
import { WorkspaceDock, type WorkspaceLayoutName } from '../layout/WorkspaceDock';
import { MenuBar } from '../layout/MenuBar';
import { StatusBar } from '../layout/StatusBar';
import { EditorDocumentTabs } from '../editors/EditorDocumentTabs';
import { EditorHost, EditorToolbarHost } from '../editors/EditorHost';
import { createEditorRegistry } from '../editors/editorRegistry';
import { useEditorDocuments } from '../editors/editorDocuments';
import type { EditorDocument } from '../editors/editorTypes';
import { LevelEditor } from '../editors/level/LevelEditor';
import { LevelEditorToolbar } from '../editors/level/LevelEditorToolbar';
import { flattenScene } from '../services/editorHostTypes';
import type { AssetItem, ConsoleEvent, ProjectSnapshot, SceneEntity } from '../services/editorHostTypes';
import { UiIconButton, UiPanel, UiTab, UiTabs, UiTreeRow } from '../ui';
import { ViewportPanel } from '../viewport/ViewportPanel';
import { WorldEnvironmentInspector } from '../environment/WorldEnvironmentInspector';
import type { HostWorldEnvironment } from '../environment/environmentTypes';
import { InspectorPanel as DataDrivenInspector } from '../inspector/InspectorPanel';
import type { HostProjectComponentSchema } from '../inspector/componentSchemas';
import type { HostEntityId, HostResponse, InspectorEntitySnapshot } from '../inspector/inspectorTypes';
import {
  aggregateInspectorSnapshots,
  hostEntityKey,
  parseSelectedEntitySnapshot,
  quaternionToEulerDegrees,
} from '../inspector/inspectorTypes';
import { ProfilerPanel } from '../profiler/ProfilerPanel';
import type { ProfilerSnapshot } from '../profiler/ProfilerPanel';
import { TerrainToolsPanel } from '../terrain/TerrainToolsPanel';
import type { TerrainToolState } from '../terrain/TerrainToolsPanel';
import { CreateTerrainDialog } from '../terrain/CreateTerrainDialog';
import { ConsolePanel } from '../console/ConsolePanel';
import { BuildOutputPanel } from '../buildOutput/BuildOutputPanel';
import type { ArcBuildRequest, ArcBuildSnapshot } from '../../../common/buildTypes';
import { AiGatewayApprovalPrompt, AiGatewayPanel } from '../ai/AiGatewayPanel';
import type { ArcAiGatewayStatus } from '../../../preload/preload';
import { RenderGraphPanel } from '../renderGraph/RenderGraphPanel';
import { ShaderEditorPanel } from '../shader/ShaderEditorPanel';
import { LightingPanel } from '../lighting/LightingPanel';
import { SearchPanel } from '../search/SearchPanel';
import { SettingsDialog } from '../settings/SettingsDialog';
import { VersionControlPanel } from '../versionControl/VersionControlPanel';
import { ContentBrowserPanel as ContentBrowserV2 } from '../content/ContentBrowserPanel';

import './workbench.css';
import '../editors/editorShell.css';

export type HostSceneEntity = {
  entity: HostEntityId;
  guid: string;
  parentGuid: string;
  siblingOrder: number;
  name: string;
  kind: 'camera' | 'light' | 'environment' | 'mesh' | 'primitive' | 'imported' | 'unknown';
  active: boolean;
  selected: boolean;
  documentGuid?: string;
  editorFolder?: string;
  collection?: string;
  layer?: string;
  locked?: boolean;
  visible?: boolean;
  pickable?: boolean;
  transform?: {
    position: [number, number, number];
    rotation: [number, number, number, number];
    scale: [number, number, number];
  } | null;
  prefabOverrideCount?: number;
};

type HostSceneSnapshot = {
  sceneGuid: string;
  sceneName: string;
  activeScenePath: string;
  dirty: boolean;
  canUndo: boolean;
  canRedo: boolean;
  undoLabel: string;
  redoLabel: string;
  entities: HostSceneEntity[];
};

type BasicEntityKind = 'empty' | 'plane' | 'cube' | 'sphere' | 'cylinder' | 'cone' | 'capsule' | 'terrain';

type SceneDocumentState = Omit<HostSceneSnapshot, 'entities'>;

type WorkspaceDocument = {
  guid: string;
  name: string;
  path: string;
  dirty: boolean;
  recovered: boolean;
  readOnly: boolean;
  active: boolean;
  pinned: boolean;
  entityCount: number;
  revision: number;
};

type WorkspaceDocumentsSnapshot = {
  activeDocument: string;
  documents: WorkspaceDocument[];
};

type HostRuntimeSnapshot = {
  state: 'stopped' | 'running' | 'paused' | 'faulted';
  tickId: number;
  revision: number;
  discardedTicks: number;
  timeScale: number;
  interpolationAlpha: number;
  worldCount: number;
};

type HostAssetSnapshot = {
  guid: string;
  path: string;
  scope: 'builtin' | 'project' | 'user' | 'organization';
  readOnly: boolean;
  kind: AssetItem['kind'] | 'environment' | 'unknown';
  typeId: string;
  importerId: string;
  state: 'unknown' | 'queued' | 'importing' | 'ready' | 'stale' | 'failed';
  residency: 'metadata' | 'source' | 'derived' | 'cpu' | 'device';
  generation: number;
  strongReferences: number;
  pins: number;
  diagnostic: string;
  dependencies: string[];
  reverseDependencies: string[];
  imported: boolean;
  importRunning: boolean;
  width: number;
  height: number;
  textureFormat: string;
  mipCount: number;
  tileCount: number;
  streamingMode: 'resident' | 'streamed_mips' | 'virtual_tiles';
  settingsVersion: number;
  artifactSize: number;
  streamingEligibilityError: string;
};

type HostProjectAssetsSnapshot = {
  projectName: string;
  projectRoot: string;
  assetRoot: string;
  cacheRoot: string;
  cacheLocalBytes: number;
  cacheLocalHits: number;
  cacheLocalMisses: number;
  cacheSharedHits: number;
  cacheSharedMisses: number;
  cacheCorruptEntries: number;
  cacheEvictions: number;
  cacheHitRate: number;
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
  viewportMode: 'unavailable',
};

const sceneKindFromHost = (kind: HostSceneEntity['kind']): SceneEntity['kind'] => {
  if (kind === 'camera' || kind === 'light' || kind === 'environment' || kind === 'mesh') {
    return kind;
  }
  return 'mesh';
};

const assetKindFromHost = (kind: HostAssetSnapshot['kind']): AssetItem['kind'] => {
  if (kind === 'environment') return 'texture';
  if (
    kind === 'material' ||
    kind === 'texture' ||
    kind === 'shader' ||
    kind === 'mesh' ||
    kind === 'prefab' ||
    kind === 'folder'
  ) {
    return kind;
  }
  return 'scene';
};

const assetNameFromPath = (value: string) => value.split(/[\\/]/).pop() || value;

const timestamp = () => new Date().toLocaleTimeString([], { hour12: false });

const sceneRootId = 'scene-root';

const isEditorOnlyHostEntity = (entity: HostSceneEntity) => entity.name.toLocaleLowerCase() === 'editor camera';

const sceneRootEntity = (children: SceneEntity[]): SceneEntity => ({
  id: sceneRootId,
  name: 'Scene',
  kind: 'folder',
  active: true,
  children,
});

export const buildSceneTree = (entities: HostSceneEntity[]): SceneEntity[] => {
  const byGuid = new Map<string, SceneEntity>();
  for (const entity of entities) {
    byGuid.set(entity.guid, {
      id: hostEntityKey(entity.entity),
      guid: entity.guid,
      name: entity.name,
      kind: sceneKindFromHost(entity.kind),
      active: entity.active,
      documentGuid: entity.documentGuid,
      editorFolder: entity.editorFolder,
      collection: entity.collection,
      layer: entity.layer,
      locked: entity.locked,
      visible: entity.visible ?? true,
      pickable: entity.pickable ?? true,
      transform: entity.transform
        ? {
            position: {
              x: entity.transform.position[0],
              y: entity.transform.position[1],
              z: entity.transform.position[2],
            },
            rotation: quaternionToEulerDegrees({
              x: entity.transform.rotation[0],
              y: entity.transform.rotation[1],
              z: entity.transform.rotation[2],
              w: entity.transform.rotation[3],
            }),
            scale: { x: entity.transform.scale[0], y: entity.transform.scale[1], z: entity.transform.scale[2] },
          }
        : undefined,
      prefabOverrideCount: entity.prefabOverrideCount ?? 0,
      children: [],
    });
  }
  const roots: Array<{ order: number; entity: SceneEntity }> = [];
  const childrenByParent = new Map<string, Array<{ order: number; entity: SceneEntity }>>();
  for (const source of entities) {
    const entity = byGuid.get(source.guid)!;
    if (!source.parentGuid || !byGuid.has(source.parentGuid)) {
      roots.push({ order: source.siblingOrder, entity });
      continue;
    }
    entity.parentId = byGuid.get(source.parentGuid)?.id;
    const siblings = childrenByParent.get(source.parentGuid) ?? [];
    siblings.push({ order: source.siblingOrder, entity });
    childrenByParent.set(source.parentGuid, siblings);
  }
  for (const [parentGuid, entries] of childrenByParent) {
    const parent = byGuid.get(parentGuid);
    if (parent) parent.children = entries.sort((a, b) => a.order - b.order).map((entry) => entry.entity);
  }
  return roots.sort((a, b) => a.order - b.order).map((entry) => entry.entity);
};

type HostEventLike = {
  type: string;
  entity?: HostEntityId;
};

export type HostEventRefreshAction = 'none' | 'selection' | 'selected' | 'hierarchy' | 'all';

const validHostEntity = (entity: HostEntityId | undefined): entity is HostEntityId =>
  Boolean(entity && entity.index !== 0xffffffff);

export const classifyHostEventRefresh = (event: HostEventLike, selectedEntityId: string): HostEventRefreshAction => {
  if (event.type === 'entity.selected') {
    const nextSelection = validHostEntity(event.entity) ? hostEntityKey(event.entity) : '';
    return nextSelection === selectedEntityId ? 'none' : 'all';
  }
  if (event.type === 'component.changed') {
    if (!validHostEntity(event.entity)) return 'none';
    return hostEntityKey(event.entity) === selectedEntityId ? 'selected' : 'hierarchy';
  }
  if (event.type === 'terrain.strokeCommitted') {
    if (!validHostEntity(event.entity)) return 'none';
    return hostEntityKey(event.entity) === selectedEntityId ? 'selected' : 'hierarchy';
  }
  if (
    event.type === 'scene.changed' ||
    event.type === 'entity.created' ||
    event.type === 'entity.deleted' ||
    event.type === 'project.opened' ||
    event.type === 'project.moduleReloaded' ||
    event.type === 'project.closed' ||
    event.type === 'asset.changed'
  )
    return 'all';
  return 'none';
};

const translationSnapOptions = [0.05, 0.1, 0.25, 0.5, 1] as const;
const rotationSnapOptions = [5, 10, 15, 30, 45, 90] as const;
const scaleSnapOptions = [0.05, 0.1, 0.25, 0.5] as const;
const timeScaleOptions = [0.25, 0.5, 1, 2, 4] as const;
const nextSnapOption = (options: readonly number[], current: number) =>
  options[(options.indexOf(current) + 1) % options.length];

const parseHostEntityId = (id: string): HostEntityId | null => {
  const [index, generation] = id.split(':').map((part) => Number.parseInt(part, 10));
  if (!Number.isInteger(index) || !Number.isInteger(generation)) {
    return null;
  }
  return { index, generation };
};

const terrainToolStateFromSnapshot = (snapshot: InspectorEntitySnapshot): TerrainToolState | null => {
  if (!snapshot.terrain) return null;
  return {
    entity: snapshot.entity,
    active: true,
    hoverVisible: false,
    tool: snapshot.terrain.brushTool,
    radius: snapshot.terrain.brushRadius,
    strength: snapshot.terrain.brushStrength,
    falloff: snapshot.terrain.brushFalloff,
    activeLayer: snapshot.terrain.activeLayer,
  };
};

export function Workbench({ onProjectClosed }: { onProjectClosed?: () => void } = {}) {
  const { layout, setLayout, resetLayout } = useWorkbenchLayout();
  const [startupState, setStartupState] = useState<StartupState | null>(null);
  const [project, setProject] = useState<ProjectSnapshot | null>(null);
  const [hostConsoleEvents, setHostConsoleEvents] = useState<ConsoleEvent[]>([]);
  const [consoleLocked, setConsoleLocked] = useState(true);
  const [clearedConsoleIds, setClearedConsoleIds] = useState<ReadonlySet<string>>(new Set());
  const [aiGatewayStatus, setAiGatewayStatus] = useState<ArcAiGatewayStatus | null>(null);
  const [selectedEntityId, setSelectedEntityId] = useState('');
  const [selectedEntityIds, setSelectedEntityIds] = useState<ReadonlySet<string>>(new Set());
  const selectedEntityIdRef = useRef(selectedEntityId);
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null);
  const [assetCache, setAssetCache] = useState<HostProjectAssetsSnapshot | null>(null);
  const [selectedSnapshot, setSelectedSnapshot] = useState<InspectorEntitySnapshot | null>(null);
  const [projectComponentSchemas, setProjectComponentSchemas] = useState<HostProjectComponentSchema[]>([]);
  const [buildSnapshot, setBuildSnapshot] = useState<ArcBuildSnapshot | null>(null);
  const [selectedSnapshotLoading, setSelectedSnapshotLoading] = useState(false);
  const selectedSnapshotRevision = useRef(0);
  const hostEventRefreshTimer = useRef<number | null>(null);
  const hostEventRefreshMode = useRef<'none' | 'selected' | 'hierarchy' | 'all'>('none');
  const [worldEnvironment, setWorldEnvironment] = useState<HostWorldEnvironment | null>(null);
  const [documentState, setDocumentState] = useState<SceneDocumentState>({
    sceneGuid: '',
    sceneName: 'Untitled',
    activeScenePath: '',
    dirty: true,
    canUndo: false,
    canRedo: false,
    undoLabel: '',
    redoLabel: '',
  });
  const [workspaceDocuments, setWorkspaceDocuments] = useState<WorkspaceDocumentsSnapshot>({
    activeDocument: '',
    documents: [],
  });
  const {
    documents: editorDocuments,
    activeDocumentId,
    activeDocument,
    syncSingletonDocument,
    activateDocument,
  } = useEditorDocuments();
  const [activeTool, setActiveTool] = useState<'select' | 'translate' | 'rotate' | 'scale' | 'terrain'>('translate');
  const [terrainToolState, setTerrainToolState] = useState<TerrainToolState | null>(null);
  const [coordinateSpace, setCoordinateSpace] = useState<'world' | 'local'>('world');
  const [snapping, setSnapping] = useState(false);
  const [translationSnap, setTranslationSnap] = useState(0.25);
  const [rotationSnap, setRotationSnap] = useState(15);
  const [scaleSnap, setScaleSnap] = useState(0.1);
  const [lastCommand, setLastCommand] = useState('Workbench ready');
  const [viewportGridVisible, setViewportGridVisible] = useState(true);
  const [viewportFocused, setViewportFocused] = useState(false);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [createTerrainOpen, setCreateTerrainOpen] = useState(false);
  const [requestedWorkspaceLayout, setRequestedWorkspaceLayout] = useState<WorkspaceLayoutName | 'Reset' | null>(null);
  const [requestedWorkspacePanel, setRequestedWorkspacePanel] = useState<WorkbenchPanelId | null>(null);
  const keybindings = useRef(new KeybindingService());
  const [profilerSamples, setProfilerSamples] = useState<ProfilerSnapshot[]>([]);
  const [runtimeState, setRuntimeState] = useState<HostRuntimeSnapshot>({
    state: 'stopped',
    tickId: 0,
    revision: 0,
    discardedTicks: 0,
    timeScale: 1,
    interpolationAlpha: 0,
    worldCount: 0,
  });
  const runtimeRevision = useRef(0);

  useEffect(() => {
    const workspaceDocument =
      workspaceDocuments.documents.find((entry) => entry.guid === workspaceDocuments.activeDocument) ??
      workspaceDocuments.documents[0] ??
      null;
    const levelDocument: EditorDocument | null = workspaceDocument
      ? {
          id: `level:${workspaceDocument.guid || workspaceDocument.path || 'active'}`,
          kind: 'level',
          title: workspaceDocument.name || 'Untitled',
          path: workspaceDocument.path,
          dirty: workspaceDocument.dirty,
          readOnly: workspaceDocument.readOnly,
          recovered: workspaceDocument.recovered,
        }
      : null;
    syncSingletonDocument('level', levelDocument);
  }, [syncSingletonDocument, workspaceDocuments]);

  const acceptRuntimeSnapshot = useCallback((value: unknown) => {
    if (!value || typeof value !== 'object') return;
    const candidate = value as Partial<HostRuntimeSnapshot>;
    if (
      candidate.state !== 'stopped' &&
      candidate.state !== 'running' &&
      candidate.state !== 'paused' &&
      candidate.state !== 'faulted'
    )
      return;
    const revision = Number(candidate.revision ?? 0);
    if (!Number.isFinite(revision) || revision < runtimeRevision.current) return;
    runtimeRevision.current = revision;
    setRuntimeState({
      state: candidate.state,
      tickId: Number(candidate.tickId ?? 0),
      revision,
      discardedTicks: Number(candidate.discardedTicks ?? 0),
      timeScale: Number(candidate.timeScale ?? 1),
      interpolationAlpha: Number(candidate.interpolationAlpha ?? 0),
      worldCount: Number(candidate.worldCount ?? 0),
    });
  }, []);

  const loadAssetThumbnail = useCallback(
    async (path: string): Promise<string | null> => {
      if (!startupState?.engineHostConnected || !window.arc?.host) return null;
      const response = (await window.arc.host.query('asset.thumbnail', {
        path,
        maxSize: 128,
      })) as HostResponse<HostAssetThumbnailSnapshot>;
      return response.succeeded && response.payload?.dataUrl ? response.payload.dataUrl : null;
    },
    [startupState?.engineHostConnected],
  );

  const refreshTerrainToolState = useCallback(async () => {
    if (!window.arc?.host) return;
    const response = (await window.arc.host.query('terrain.toolState')) as HostResponse<TerrainToolState>;
    if (
      response.succeeded &&
      response.payload &&
      hostEntityKey(response.payload.entity) === selectedEntityIdRef.current
    )
      setTerrainToolState(response.payload);
  }, []);

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
    });
  }, []);

  useEffect(() => {
    if (!window.arc?.aiGateway) return;
    void window.arc.aiGateway.status().then(setAiGatewayStatus);
    return window.arc.aiGateway.onStatus(setAiGatewayStatus);
  }, []);

  useEffect(() => {
    if (!window.arc?.build) return;
    void window.arc.build.snapshot().then(setBuildSnapshot);
    return window.arc.build.onState(setBuildSnapshot);
  }, []);

  useEffect(
    () =>
      window.arc?.host?.onEvent?.((event) => {
        if (event.type === 'profiler.snapshot' && event.payload && typeof event.payload === 'object') {
          const sample = event.payload as ProfilerSnapshot;
          setProfilerSamples((current) => [...current, sample].slice(-3000));
          return;
        }
        if (
          (event.type === 'runtime.stateChanged' ||
            event.type === 'runtime.tickCompleted' ||
            event.type === 'runtime.fault') &&
          event.payload
        ) {
          acceptRuntimeSnapshot(event.payload);
          setLastCommand(event.message || event.type);
          return;
        }
        if (event.type === 'terrain.toolChanged' && event.payload) {
          const next = event.payload as TerrainToolState;
          if (hostEntityKey(next.entity) === selectedEntityIdRef.current) setTerrainToolState(next);
          setLastCommand(event.message || event.type);
          return;
        }
        setLastCommand(event.message || event.type);
        if (event.payload && typeof event.payload === 'object' && 'tool' in event.payload) {
          const payload = event.payload as {
            tool?: unknown;
            coordinateSpace?: unknown;
            snapping?: unknown;
            translationSnap?: unknown;
            rotationSnapDegrees?: unknown;
            scaleSnap?: unknown;
          };
          const tool = String(payload.tool);
          if (tool === 'select' || tool === 'translate' || tool === 'rotate' || tool === 'scale' || tool === 'terrain')
            setActiveTool(tool);
          if (payload.coordinateSpace === 'world' || payload.coordinateSpace === 'local')
            setCoordinateSpace(payload.coordinateSpace);
          if (typeof payload.snapping === 'boolean') setSnapping(payload.snapping);
          if (typeof payload.translationSnap === 'number') setTranslationSnap(payload.translationSnap);
          if (typeof payload.rotationSnapDegrees === 'number') setRotationSnap(payload.rotationSnapDegrees);
          if (typeof payload.scaleSnap === 'number') setScaleSnap(payload.scaleSnap);
        }

        const action = classifyHostEventRefresh(event, selectedEntityIdRef.current);
        if (action === 'none') return;
        if (action === 'selection') {
          const nextSelection = validHostEntity(event.entity) ? hostEntityKey(event.entity) : '';
          selectedEntityIdRef.current = nextSelection;
          setSelectedEntityId(nextSelection);
          if (!nextSelection) {
            ++selectedSnapshotRevision.current;
            setSelectedSnapshot(null);
            return;
          }
          if (hostEventRefreshMode.current === 'hierarchy') hostEventRefreshMode.current = 'all';
          else if (hostEventRefreshMode.current !== 'all') hostEventRefreshMode.current = 'selected';
        } else if (action === 'all') {
          hostEventRefreshMode.current = 'all';
        } else if (action === 'hierarchy' && hostEventRefreshMode.current !== 'all') {
          hostEventRefreshMode.current = 'hierarchy';
        } else if (action === 'selected' && hostEventRefreshMode.current === 'none') {
          hostEventRefreshMode.current = 'selected';
        }

        if (hostEventRefreshTimer.current !== null) return;
        hostEventRefreshTimer.current = window.setTimeout(() => {
          hostEventRefreshTimer.current = null;
          const mode = hostEventRefreshMode.current;
          hostEventRefreshMode.current = 'none';
          if (mode === 'all') void refreshProjectFromHost();
          else if (mode === 'hierarchy') void refreshProjectFromHost(undefined, false);
          else if (mode === 'selected' && selectedEntityIdRef.current)
            void refreshSelectedEntity(selectedEntityIdRef.current, true);
        }, 24);
      }),
    // The host event subscription intentionally reads the latest refresh functions through render scope.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [acceptRuntimeSnapshot],
  );

  useEffect(() => {
    if (activeTool !== 'terrain') return;
    if (selectedSnapshot && !selectedSnapshot.terrain) {
      setActiveTool('select');
      if (startupState?.engineHostConnected)
        void window.arc.host.command('viewport.setTool', {
          tool: 'select',
          coordinateSpace,
          snapping,
          translationSnap,
          rotationSnapDegrees: rotationSnap,
          scaleSnap,
        });
      return;
    }
    if (selectedSnapshot?.terrain) void refreshTerrainToolState();
  }, [
    activeTool,
    coordinateSpace,
    refreshTerrainToolState,
    rotationSnap,
    scaleSnap,
    selectedSnapshot,
    snapping,
    startupState?.engineHostConnected,
    translationSnap,
  ]);

  const commandContext: CommandContext = {
    editorFocused: true,
    viewportFocused,
    textInputFocused: false,
    modalOpen: commandPaletteOpen || settingsOpen || createTerrainOpen,
    playing: runtimeState.state === 'running' || runtimeState.state === 'paused',
    hasSelection: Boolean(selectedEntityId),
    canUndo: documentState.canUndo,
    canRedo: documentState.canRedo,
    projectOpen: Boolean(project),
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const textInputFocused = Boolean(target?.matches('input, textarea, select, [contenteditable="true"]'));
      if (textInputFocused && !(event.ctrlKey && event.key.toLocaleLowerCase() === 'k')) return;
      const match = keybindings.current.match(event, { ...commandContext, textInputFocused });
      if (!match) return;
      event.preventDefault();
      if (!match.chordPending) void runCommand(match.command);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  });

  const refreshSelectedEntity = async (
    entityId = selectedEntityId,
    connected = startupState?.engineHostConnected ?? false,
  ) => {
    const requestRevision = ++selectedSnapshotRevision.current;
    if (!connected || !window.arc?.host) {
      if (requestRevision === selectedSnapshotRevision.current) setSelectedSnapshot(null);
      return;
    }
    const replacingSelection = !selectedSnapshot || hostEntityKey(selectedSnapshot.entity) !== entityId;
    if (replacingSelection) setSelectedSnapshotLoading(true);
    try {
      const response = (await window.arc.host.query('entity.selected')) as HostResponse<unknown>;
      if (requestRevision !== selectedSnapshotRevision.current) return;
      if (!response.succeeded || !response.payload) {
        if (replacingSelection) setSelectedSnapshot(null);
        setLastCommand(response.error || 'Could not read selected entity');
        return;
      }
      let next = parseSelectedEntitySnapshot(response.payload);
      if ((next.selectionCount ?? 1) > 1 && next.selectedGuids?.length) {
        const responses = await Promise.all(
          next.selectedGuids.map(
            (guid) => window.arc.host.query('gateway.entity', { guid }) as Promise<HostResponse<unknown>>,
          ),
        );
        if (requestRevision !== selectedSnapshotRevision.current) return;
        const snapshots = responses
          .filter((entry) => entry.succeeded && entry.payload)
          .map((entry) => parseSelectedEntitySnapshot(entry.payload));
        if (snapshots.length !== next.selectedGuids.length)
          throw new Error('Selection changed while the aggregate Inspector snapshot was being built');
        const primary = snapshots.find((entry) => hostEntityKey(entry.entity) === entityId) ?? next;
        next = aggregateInspectorSnapshots(primary, snapshots);
      }
      setSelectedSnapshot(next);
    } catch (error) {
      if (requestRevision !== selectedSnapshotRevision.current) return;
      if (replacingSelection) setSelectedSnapshot(null);
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
      const response = (await window.arc.host.query('environment.state', {
        entity,
      })) as HostResponse<HostWorldEnvironment>;
      setWorldEnvironment(response.succeeded && response.payload ? response.payload : null);
    } catch {
      setWorldEnvironment(null);
    }
  };

  const runCommand = async (command: CommandId) => {
    const registration = commandRegistry[command];
    if (registration.enabled && !registration.enabled(commandContext)) {
      setLastCommand(registration.disabledReason?.(commandContext) ?? `${registration.label} is unavailable`);
      return;
    }

    if (command === 'view.commandPalette') {
      setCommandPaletteOpen(true);
      return;
    }
    if (command === 'settings.open') {
      setCommandPaletteOpen(false);
      setSettingsOpen(true);
      setLayout((current) => ({ ...current, activityExpanded: false }));
      return;
    }
    if (command === 'layout.reset') {
      resetLayout();
      setRequestedWorkspaceLayout('Reset');
      setLastCommand('Layout reset');
      return;
    }

    if (command === 'layout.levelDesign' || command === 'layout.materials' || command === 'layout.profiling') {
      const name: WorkspaceLayoutName =
        command === 'layout.materials' ? 'Materials' : command === 'layout.profiling' ? 'Profiling' : 'Level Design';
      setRequestedWorkspaceLayout(name);
      setLastCommand(`Switched to ${name} layout`);
      return;
    }

    if (command === 'project.close') {
      try {
        const result = await window.arc.projects.close();
        if (!result.succeeded) {
          setLastCommand(result.error || 'Project close cancelled');
          return;
        }
        onProjectClosed?.();
      } catch (error) {
        setLastCommand(error instanceof Error ? error.message : String(error));
      }
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
        setLastCommand(
          `${append ? 'Imported' : 'Opened'} ${assetNameFromPath(result.filePath ?? 'scene')} (${count} entities)`,
        );
      } catch (error) {
        setLastCommand(error instanceof Error ? error.message : String(error));
      }
      return;
    }

    if (startupState?.engineHostConnected && window.arc?.host) {
      try {
        let response: HostResponse | undefined;
        if (command === 'file.new') {
          response = (await window.arc.host.command('scene.new', { name: 'Untitled' })) as HostResponse;
        } else if (command === 'file.save' || command === 'assets.saveAll') {
          if (!documentState.activeScenePath) {
            const result = await window.arc.dialog.saveScene();
            if (result.canceled) return setLastCommand('Save canceled');
            response = result.response as HostResponse;
          } else {
            response = (await window.arc.host.command('scene.save')) as HostResponse;
          }
        } else if (command === 'file.saveAs') {
          const result = await window.arc.dialog.saveScene();
          if (result.canceled) return setLastCommand('Save canceled');
          response = result.response as HostResponse;
        } else if (command === 'edit.undo') {
          response = (await window.arc.host.command('history.undo')) as HostResponse;
        } else if (command === 'edit.redo') {
          response = (await window.arc.host.command('history.redo')) as HostResponse;
        } else if (command === 'entity.duplicate' && selectedSnapshot) {
          response = (await window.arc.host.command('entity.duplicate', {
            entity: selectedSnapshot.entity,
          })) as HostResponse;
        } else if (command === 'entity.delete' && selectedSnapshot) {
          response = (await window.arc.host.command('entity.delete', {
            entity: selectedSnapshot.entity,
          })) as HostResponse;
        } else if (command === 'scene.play') {
          response = (await window.arc.host.command('runtime.resume')) as HostResponse<HostRuntimeSnapshot>;
        } else if (command === 'scene.pause') {
          response = (await window.arc.host.command('runtime.pause')) as HostResponse<HostRuntimeSnapshot>;
        } else if (command === 'scene.stop') {
          response = (await window.arc.host.command('runtime.stop')) as HostResponse<HostRuntimeSnapshot>;
        } else if (command === 'scene.step') {
          response = (await window.arc.host.command('runtime.step', { ticks: 1 })) as HostResponse<HostRuntimeSnapshot>;
        } else if (command.startsWith('viewport.')) {
          if (command === 'viewport.frameSelected') {
            await window.arc.viewport.cameraInput({ focusSelected: true });
            return setLastCommand('Framed selected entity');
          }
          const tool = command.slice('viewport.'.length) as typeof activeTool;
          if (
            tool === 'select' ||
            tool === 'translate' ||
            tool === 'rotate' ||
            tool === 'scale' ||
            tool === 'terrain'
          ) {
            response = (await window.arc.host.command('viewport.setTool', {
              tool,
              coordinateSpace,
              snapping,
              translationSnap,
              rotationSnapDegrees: rotationSnap,
              scaleSnap,
            })) as HostResponse;
            if (response?.succeeded) setActiveTool(tool);
          }
        }
        if (response) {
          if (command.startsWith('scene.') && response.succeeded) acceptRuntimeSnapshot(response.payload);
          setLastCommand(response.succeeded ? `${command} completed` : response.error || `${command} failed`);
          const responsePath =
            response.payload && typeof response.payload === 'object' && 'path' in response.payload
              ? String((response.payload as { path?: unknown }).path ?? '')
              : undefined;
          const refreshesScene =
            command === 'file.new' ||
            command === 'file.save' ||
            command === 'file.saveAs' ||
            command === 'edit.undo' ||
            command === 'edit.redo' ||
            command === 'entity.duplicate' ||
            command === 'entity.delete';
          if (response.succeeded && refreshesScene) await refreshProjectFromHost(responsePath);
          if (response.succeeded && command === 'viewport.terrain')
            setLayout((current) => ({ ...current, leftVisible: true }));
          return;
        }
      } catch (error) {
        setLastCommand(error instanceof Error ? error.message : String(error));
        return;
      }
    }

    setLastCommand(
      startupState?.engineHostConnected
        ? `${command} is not implemented by the native host`
        : 'Native editor host is unavailable',
    );
  };

  const refreshProjectFromHost = async (activeScene?: string, refreshSelection = true) => {
    if (!window.arc?.host) {
      return;
    }

    const [sceneResponse, assetsResponse, documentsResponse, schemasResponse] = await Promise.all([
      window.arc.host.query('scene.hierarchy') as Promise<HostResponse<HostSceneSnapshot>>,
      window.arc.host.query('project.assets') as Promise<HostResponse<HostProjectAssetsSnapshot>>,
      window.arc.host.query('workspace.documents') as Promise<HostResponse<WorkspaceDocumentsSnapshot>>,
      window.arc.host.query('gateway.componentSchemas') as Promise<
        HostResponse<{ components: HostProjectComponentSchema[] }>
      >,
    ]);

    if (!sceneResponse.succeeded || !sceneResponse.payload) {
      return;
    }

    const scenePayload = sceneResponse.payload;
    const hostEntities = scenePayload.entities.filter((entity) => !isEditorOnlyHostEntity(entity));
    const scene = buildSceneTree(hostEntities);
    const { entities: _entities, ...nextDocumentState } = scenePayload;
    void _entities;
    setDocumentState(nextDocumentState);
    if (documentsResponse.succeeded && documentsResponse.payload) setWorkspaceDocuments(documentsResponse.payload);
    if (schemasResponse.succeeded && schemasResponse.payload)
      setProjectComponentSchemas(schemasResponse.payload.components.filter((component) => component.projectComponent));

    const hostAssets = assetsResponse.succeeded && assetsResponse.payload ? assetsResponse.payload : null;
    setAssetCache(hostAssets);
    const assets =
      hostAssets?.assets.map((asset): AssetItem => ({
        id: asset.guid || asset.path,
        name: assetNameFromPath(asset.path),
        path: asset.path,
        scope: asset.scope,
        readOnly: asset.readOnly,
        kind: assetKindFromHost(asset.kind),
        status: asset.state === 'unknown' ? 'missing' : asset.state,
        guid: asset.guid,
        typeId: asset.typeId,
        importerId: asset.importerId,
        residency: asset.residency,
        generation: asset.generation,
        diagnostic: asset.diagnostic,
        dependencies: asset.dependencies,
        reverseDependencies: asset.reverseDependencies,
        width: asset.width,
        height: asset.height,
        textureFormat: asset.textureFormat,
        mipLevels: asset.mipCount,
        tileCount: asset.tileCount,
        streamingMode: asset.streamingMode,
        settingsVersion: asset.settingsVersion,
        artifactSize: asset.artifactSize,
        streamingEligibilityError: asset.streamingEligibilityError,
      })) ??
      project?.assets ??
      [];

    const selected = hostEntities.find((entity) => entity.selected);
    setSelectedEntityIds(
      new Set(hostEntities.filter((entity) => entity.selected).map((entity) => hostEntityKey(entity.entity))),
    );
    if (selected && refreshSelection) {
      const selectedKey = hostEntityKey(selected.entity);
      selectedEntityIdRef.current = selectedKey;
      setSelectedEntityId(selectedKey);
      await refreshSelectedEntity(selectedKey, true);
    } else if (refreshSelection) {
      selectedEntityIdRef.current = '';
      setSelectedEntityId('');
      setSelectedEntityIds(new Set());
      ++selectedSnapshotRevision.current;
      setSelectedSnapshot(null);
    }
    const environmentEntity = hostEntities.find((entity) => entity.kind === 'environment');
    if (environmentEntity) await refreshWorldEnvironment(hostEntityKey(environmentEntity.entity));
    if (activeScene) {
      setSelectedAssetId(activeScene);
    }

    setProject((current) => ({
      ...(current ?? {
        name: hostAssets?.projectName || 'Project',
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
      name: hostAssets?.projectName || current?.name || 'Project',
      root: hostAssets?.projectRoot || current?.root || '',
      assetRoot: hostAssets?.assetRoot || current?.assetRoot || '',
      activeScene: activeScene ?? scenePayload.activeScenePath ?? current?.activeScene ?? '',
      scene,
      assets,
      console: current?.console ?? [],
    }));
  };

  const reconnectHost = async () => {
    setLastCommand('Reconnecting native editor host...');
    try {
      const state = await window.arc.host.reconnect();
      setStartupState(state);
      if (!state.engineHostConnected) {
        setProject(null);
        setLastCommand(state.hostError || 'Native editor host is unavailable');
        return;
      }
      const runtimeResponse = (await window.arc.host.query('runtime.state')) as HostResponse<HostRuntimeSnapshot>;
      if (runtimeResponse.succeeded) acceptRuntimeSnapshot(runtimeResponse.payload);
      await refreshProjectFromHost();
      setLastCommand('Native editor host reconnected');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStartupState({ ...fallbackStartupState, hostError: message });
      setProject(null);
      setLastCommand(message);
    }
  };

  useEffect(() => {
    const startup = window.arc?.getStartupState?.() ?? Promise.resolve(fallbackStartupState);
    void startup
      .then(async (state) => {
        setStartupState(state);
        if (state.engineHostConnected) {
          const runtimeResponse = (await window.arc.host.query('runtime.state')) as HostResponse<HostRuntimeSnapshot>;
          if (runtimeResponse.succeeded) acceptRuntimeSnapshot(runtimeResponse.payload);
          await refreshProjectFromHost();
          return;
        }
        setProject(null);
        setLastCommand(state.hostError || 'Native editor host is unavailable');
      })
      .catch((error) => {
        setStartupState(fallbackStartupState);
        setProject(null);
        setLastCommand(error instanceof Error ? error.message : String(error));
      });
    // Startup owns the first authoritative refresh; later refreshes are event-driven.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [acceptRuntimeSnapshot]);

  const selectEntity = async (entityId: string, additive = false) => {
    if (!additive && entityId === selectedEntityIdRef.current && selectedEntityIds.size === 1) return;
    const hostEntity = parseHostEntityId(entityId);
    if (startupState?.engineHostConnected && hostEntity) {
      selectedEntityIdRef.current = entityId;
      setSelectedEntityId(entityId);
      await window.arc.host.command('entity.select', { entity: hostEntity, additive, toggle: additive });
      await refreshProjectFromHost(undefined, true);
    }
  };

  const mutateHierarchyEntity = async (type: string, payload: Record<string, unknown>) => {
    if (!startupState?.engineHostConnected) return false;
    const response = (await window.arc.host.command(type, payload)) as HostResponse;
    setLastCommand(response.succeeded ? `${type} completed` : response.error || `${type} failed`);
    if (response.succeeded) await refreshProjectFromHost();
    return response.succeeded;
  };

  const renameHierarchyEntity = (entityId: string, name: string) => {
    const entity = parseHostEntityId(entityId);
    if (entity && name.trim()) void mutateHierarchyEntity('entity.rename', { entity, name: name.trim() });
  };

  const setHierarchyEntityActive = (entityId: string, active: boolean) => {
    const entity = parseHostEntityId(entityId);
    if (entity) void mutateHierarchyEntity('entity.setActive', { entity, active });
  };

  const createHierarchyEntity = (kind: BasicEntityKind) => {
    if (kind === 'terrain') {
      setCreateTerrainOpen(true);
      return;
    }
    const parent = selectedSnapshot?.entity;
    void mutateHierarchyEntity('entity.create', { kind, ...(parent ? { parent } : {}) });
  };

  const toggleViewportGrid = async () => {
    const visible = !viewportGridVisible;
    setViewportGridVisible(visible);
    try {
      const state = (await window.arc.host.query('viewport.state')) as HostResponse<{
        renderOptions?: Record<string, unknown>;
      }>;
      if (!state.succeeded || !state.payload?.renderOptions)
        throw new Error(state.error || 'Viewport state is unavailable');
      const response = (await window.arc.host.command('viewport.setRenderOptions', {
        ...state.payload.renderOptions,
        grid: visible,
      })) as HostResponse;
      if (!response.succeeded) throw new Error(response.error || 'Could not update viewport grid');
    } catch (error) {
      setViewportGridVisible(!visible);
      setLastCommand(error instanceof Error ? error.message : String(error));
    }
  };

  const createPrefabFromSelection = async () => {
    if (!selectedSnapshot || !window.arc?.dialog?.createPrefab) {
      setLastCommand('Select an entity before creating a prefab');
      return;
    }
    try {
      const result = await window.arc.dialog.createPrefab(selectedSnapshot.entity);
      if (result.canceled) return setLastCommand('Prefab creation canceled');
      const response = result.response as HostResponse | undefined;
      setLastCommand(response?.succeeded ? 'Prefab created' : response?.error || 'Prefab creation failed');
      if (response?.succeeded) await refreshProjectFromHost(undefined, true);
    } catch (error) {
      setLastCommand(error instanceof Error ? error.message : String(error));
    }
  };

  const instantiatePrefab = async (prefabPath?: string) => {
    if (!window.arc?.host) return;
    try {
      let response: HostResponse | undefined;
      if (prefabPath) {
        response = (await window.arc.host.command('prefab.instantiate', { path: prefabPath })) as HostResponse;
      } else if (window.arc.dialog?.instantiatePrefab) {
        const result = await window.arc.dialog.instantiatePrefab();
        if (result.canceled) return setLastCommand('Prefab instantiation canceled');
        response = result.response as HostResponse | undefined;
      }
      setLastCommand(response?.succeeded ? 'Prefab instantiated' : response?.error || 'Prefab instantiation failed');
      if (response?.succeeded) await refreshProjectFromHost(undefined, true);
    } catch (error) {
      setLastCommand(error instanceof Error ? error.message : String(error));
    }
  };

  const moveHierarchyEntity = (entityId: string, target: SceneEntity, mode: 'before' | 'inside' | 'after') => {
    const entity = parseHostEntityId(entityId);
    if (!entity || entityId === target.id) return;
    if (target.id === sceneRootId) {
      void mutateHierarchyEntity('entity.reparent', { entity, preserveWorld: true });
      return;
    }
    if (mode === 'inside') {
      const parent = parseHostEntityId(target.id);
      if (parent) void mutateHierarchyEntity('entity.reparent', { entity, parent, preserveWorld: true });
      return;
    }
    const allEntities = project ? flattenScene(project.scene) : [];
    const source = allEntities.find((value) => value.id === entityId);
    const siblings = target.parentId
      ? (allEntities.find((value) => value.id === target.parentId)?.children ?? [])
      : (project?.scene ?? []);
    const targetIndex = siblings.findIndex((value) => value.id === target.id);
    const beforeTarget = mode === 'before' ? target : siblings[targetIndex + 1];
    const beforeSibling = beforeTarget ? parseHostEntityId(beforeTarget.id) : null;
    if (source?.parentId === target.parentId) {
      void mutateHierarchyEntity('entity.reorder', { entity, ...(beforeSibling ? { beforeSibling } : {}) });
      return;
    }
    const parent = target.parentId ? parseHostEntityId(target.parentId) : null;
    void mutateHierarchyEntity('entity.reparent', {
      entity,
      ...(parent ? { parent } : {}),
      ...(beforeSibling ? { beforeSibling } : {}),
      preserveWorld: true,
    });
  };

  const updateWorldEnvironment = (next: HostWorldEnvironment) => {
    if (!startupState?.engineHostConnected) {
      setLastCommand('Native editor host is unavailable');
      return;
    }
    setWorldEnvironment(next);
    void window.arc.host
      .command('environment.update', { environment: next })
      .then((response) => {
        const result = response as HostResponse;
        setLastCommand(result.succeeded ? 'World environment updated' : result.error || 'Environment update failed');
      })
      .catch((error) => setLastCommand(error instanceof Error ? error.message : String(error)));
  };

  const applyWorldEnvironmentPreset = async (preset: string) => {
    if (!worldEnvironment || !startupState?.engineHostConnected) return;
    const response = (await window.arc.host.command('environment.applyPreset', {
      entity: worldEnvironment.entity,
      preset,
    })) as HostResponse;
    setLastCommand(response.succeeded ? `Applied ${preset} environment preset` : response.error || 'Preset failed');
    if (response.succeeded) await refreshWorldEnvironment(hostEntityKey(worldEnvironment.entity));
  };

  const applyWorldEnvironmentHdri = async (path: string): Promise<boolean> => {
    if (!worldEnvironment) return false;
    if (!startupState?.engineHostConnected) {
      setLastCommand('Native editor host is unavailable');
      return false;
    }
    setWorldEnvironment({ ...worldEnvironment, hdriPath: path });
    const response = (await window.arc.host.command('environment.setHdri', {
      entity: worldEnvironment.entity,
      path,
    })) as HostResponse;
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
    if (activity) setRequestedWorkspacePanel(activity.panelId);
  };

  const setActivityExpanded = (expanded: boolean) =>
    setLayout((current) => ({ ...current, activityExpanded: expanded }));
  const updateViewportToolOptions = async (
    nextSpace: 'world' | 'local',
    nextSnapping: boolean,
    nextTranslationSnap = translationSnap,
    nextRotationSnap = rotationSnap,
    nextScaleSnap = scaleSnap,
  ) => {
    if (!startupState?.engineHostConnected) return setLastCommand('Native editor host is unavailable');
    const response = (await window.arc.host.command('viewport.setTool', {
      tool: activeTool,
      coordinateSpace: nextSpace,
      snapping: nextSnapping,
      translationSnap: nextTranslationSnap,
      rotationSnapDegrees: nextRotationSnap,
      scaleSnap: nextScaleSnap,
    })) as HostResponse;
    if (!response.succeeded) return setLastCommand(response.error || 'Viewport tool update failed');
    setCoordinateSpace(nextSpace);
    setSnapping(nextSnapping);
    setTranslationSnap(nextTranslationSnap);
    setRotationSnap(nextRotationSnap);
    setScaleSnap(nextScaleSnap);
  };

  const renderLeftPanel = (requestedPanel?: WorkbenchPanelId) => {
    if (!project) {
      return <div className="side-loading">Loading workbench data...</div>;
    }

    if ((!requestedPanel || requestedPanel === 'hierarchy') && activeTool === 'terrain' && selectedSnapshot?.terrain) {
      const selectedKey = hostEntityKey(selectedSnapshot.entity);
      const visibleTerrainState =
        terrainToolState && hostEntityKey(terrainToolState.entity) === selectedKey
          ? terrainToolState
          : terrainToolStateFromSnapshot(selectedSnapshot)!;
      return (
        <TerrainToolsPanel
          terrain={selectedSnapshot.terrain}
          state={visibleTerrainState}
          assets={project.assets}
          thumbnailProvider={loadAssetThumbnail}
          onStateChange={setTerrainToolState}
          onStatus={setLastCommand}
          command={async (type, payload) => {
            if (!startupState?.engineHostConnected)
              return {
                succeeded: false,
                error: 'Native editor host is unavailable',
              };
            return window.arc.host.command(type, payload as Record<string, unknown>) as Promise<
              HostResponse<TerrainToolState>
            >;
          }}
        />
      );
    }

    if (requestedPanel === 'hierarchy' || (!requestedPanel && layout.activeActivity === 'scene')) {
      return (
        <ExplorerPanel
          project={project}
          selectedEntityId={selectedEntityId}
          selectedEntityIds={selectedEntityIds}
          onSelectEntity={selectEntity}
          onRenameEntity={renameHierarchyEntity}
          onSetEntityActive={setHierarchyEntityActive}
          onMoveEntity={moveHierarchyEntity}
          onCreateEntity={createHierarchyEntity}
          onDuplicate={() => void runCommand('entity.duplicate')}
          onCreatePrefab={() => void createPrefabFromSelection()}
          onInstantiatePrefab={() => void instantiatePrefab()}
          onDelete={() => void runCommand('entity.delete')}
        />
      );
    }

    if (requestedPanel === 'assetExplorer' || (!requestedPanel && layout.activeActivity === 'assets')) {
      return (
        <AssetExplorerPanel project={project} selectedAssetId={selectedAssetId} onSelectAsset={setSelectedAssetId} />
      );
    }

    if (requestedPanel === 'search' || (!requestedPanel && layout.activeActivity === 'search')) {
      return (
        <SearchPanel
          entities={project.scene}
          assets={project.assets}
          onSelectEntity={(entityId) => void selectEntity(entityId)}
          onSelectAsset={(assetId) => {
            setSelectedAssetId(assetId);
            if (project.assets.find((asset) => asset.id === assetId)?.kind === 'shader')
              setLayout((current) => ({ ...current, activeCenterPanel: 'shaderEditor' }));
          }}
        />
      );
    }

    const panelId = activityRegistry.find((entry) => entry.id === layout.activeActivity)?.panelId ?? 'hierarchy';
    const panel = getPanel(panelId);
    return (
      <PlaceholderPanel
        icon={<panel.icon />}
        title={panel.title}
        text="This panel is available in the main dock area."
      />
    );
  };

  const [viewportCount, setViewportCount] = useState<1 | 2 | 3 | 4>(1);
  const [activeViewportId, setActiveViewportId] = useState('viewport-1');

  const editorRegistry = createEditorRegistry({
    level: {
      kind: 'level',
      title: 'Level Editor',
      icon: FileText,
      allowMultiple: false,
      render: (_document, context) => (
        <LevelEditor>
          <ViewportPanel
            viewportId={context.instanceId}
            project={project}
            startupState={startupState}
            onCommand={runCommand}
            onReconnect={reconnectHost}
            gridVisible={viewportGridVisible}
            onGridVisibilityChange={setViewportGridVisible}
            active={!createTerrainOpen && !settingsOpen && (context.instanceId ?? 'viewport-1') === activeViewportId}
            onFocusChange={(focused) => {
              setViewportFocused(focused);
              if (focused) setActiveViewportId(context.instanceId ?? 'viewport-1');
            }}
            onMaximizeToggle={context.onMaximizeToggle}
            onViewportLayoutChange={setViewportCount}
          />
        </LevelEditor>
      ),
      renderToolbar: () => (
        <LevelEditorToolbar
          activeTool={activeTool}
          coordinateSpace={coordinateSpace}
          snapping={snapping}
          terrainEnabled={selectedSnapshot?.terrain !== null && selectedSnapshot?.terrain !== undefined}
          translationSnap={translationSnap}
          rotationSnap={rotationSnap}
          scaleSnap={scaleSnap}
          onCommand={runCommand}
          onBuild={() => {
            setLayout((current) => ({ ...current, bottomVisible: true, activeBottomPanel: 'buildOutput' }));
            void window.arc.build
              .execute({ action: 'build' })
              .then(setBuildSnapshot)
              .catch((reason: unknown) => {
                setLastCommand(reason instanceof Error ? reason.message : String(reason));
              });
          }}
          onLayout={(name) => setRequestedWorkspaceLayout(name)}
          onPanel={(panel) => setRequestedWorkspacePanel(panel)}
          runtimeState={runtimeState.state}
          timeScale={runtimeState.timeScale}
          onCycleTimeScale={() => {
            const next = nextSnapOption(timeScaleOptions, runtimeState.timeScale);
            if (!startupState?.engineHostConnected) {
              setLastCommand('Native editor host is unavailable');
              return;
            }
            void window.arc.host.command('runtime.setTimeScale', { value: next }).then((response) => {
              const result = response as HostResponse<HostRuntimeSnapshot>;
              if (result.succeeded) acceptRuntimeSnapshot(result.payload);
              else setLastCommand(result.error || 'Could not change runtime time scale');
            });
          }}
          onToggleCoordinateSpace={() =>
            void updateViewportToolOptions(coordinateSpace === 'world' ? 'local' : 'world', snapping)
          }
          onToggleSnapping={() => void updateViewportToolOptions(coordinateSpace, !snapping)}
          onCycleTranslationSnap={() =>
            void updateViewportToolOptions(
              coordinateSpace,
              snapping,
              nextSnapOption(translationSnapOptions, translationSnap),
            )
          }
          onCycleRotationSnap={() =>
            void updateViewportToolOptions(
              coordinateSpace,
              snapping,
              translationSnap,
              nextSnapOption(rotationSnapOptions, rotationSnap),
            )
          }
          onCycleScaleSnap={() =>
            void updateViewportToolOptions(
              coordinateSpace,
              snapping,
              translationSnap,
              rotationSnap,
              nextSnapOption(scaleSnapOptions, scaleSnap),
            )
          }
        />
      ),
    },
  });

  const renderCenterPanel = (panel: WorkbenchPanelId, viewportId?: string, onMaximizeToggle?: () => void) => {
    if (panel === 'viewport') {
      return (
        <EditorHost
          document={activeDocument}
          registry={editorRegistry}
          context={{ instanceId: viewportId, onMaximizeToggle }}
        />
      );
    }

    if (panel === 'renderGraph') return <RenderGraphPanel />;
    if (panel === 'shaderEditor')
      return <ShaderEditorPanel asset={project?.assets.find((asset) => asset.id === selectedAssetId) ?? null} />;
    return <div className="tool-empty">Panel unavailable.</div>;
  };

  const renderRightPanel = (panel: WorkbenchPanelId) => {
    if (panel === 'inspector') {
      return (
        <DataDrivenInspector
          command={async (type, payload, edit) => {
            if (!startupState?.engineHostConnected)
              return { succeeded: false, error: 'Native editor host is unavailable' };
            return window.arc.host.command(type, payload, edit) as Promise<HostResponse>;
          }}
          loading={selectedSnapshotLoading}
          snapshot={selectedSnapshot}
          coordinateSpace={coordinateSpace}
          onCoordinateSpaceChange={(space) => void updateViewportToolOptions(space, snapping)}
          scene={project?.scene ?? []}
          assets={project?.assets ?? []}
          thumbnailProvider={loadAssetThumbnail}
          projectSchemas={projectComponentSchemas}
          onStatus={setLastCommand}
          refresh={async () => {
            if (startupState?.engineHostConnected) await refreshSelectedEntity(selectedEntityId, true);
          }}
        />
      );
    }
    if (panel === 'worldSettings') {
      return (
        <WorldSettingsPanel
          environment={worldEnvironment}
          onEnvironmentChange={updateWorldEnvironment}
          assets={project?.assets ?? []}
          thumbnailProvider={loadAssetThumbnail}
          onEnvironmentPreset={applyWorldEnvironmentPreset}
          onEnvironmentHdri={applyWorldEnvironmentHdri}
        />
      );
    }
    return <LightingPanel entities={project?.scene ?? []} onSelect={(id) => void selectEntity(id)} />;
  };

  const renderBottomPanel = (panel: WorkbenchPanelId) => {
    if (panel === 'contentBrowser') {
      return (
        <ContentBrowserV2
          project={project}
          cache={assetCache}
          selectedAssetId={selectedAssetId}
          onSelectAsset={setSelectedAssetId}
          onCommand={runCommand}
          onInstantiatePrefab={(path) => void instantiatePrefab(path)}
          onAssetAction={async (type, guid) => {
            const response = (await window.arc.host.command(type, { guid })) as HostResponse;
            setLastCommand(response.succeeded ? `${type} completed` : response.error || `${type} failed`);
            if (response.succeeded) await refreshProjectFromHost(undefined, false);
          }}
          onTextureStreamingMode={async (guid, mode) => {
            const response = (await window.arc.host.command('asset.setTextureStreamingMode', {
              guid,
              mode,
            })) as HostResponse;
            setLastCommand(
              response.succeeded
                ? `Texture mode changed to ${mode}`
                : response.error || 'Texture mode change failed',
            );
            if (response.succeeded) await refreshProjectFromHost(undefined, false);
          }}
          thumbnailProvider={loadAssetThumbnail}
        />
      );
    }

    if (panel === 'console') {
      const events = [...(project?.console ?? []), ...hostConsoleEvents];
      return (
        <ConsolePanel
          events={events}
          clearedIds={clearedConsoleIds}
          locked={consoleLocked}
          onClear={(current) =>
            setClearedConsoleIds((cleared) => {
              const next = new Set(cleared);
              current.forEach((event) => next.add(event.id));
              return next;
            })
          }
          onLockedChange={setConsoleLocked}
        />
      );
    }

    if (panel === 'buildOutput') {
      return (
        <BuildOutputPanel
          snapshot={buildSnapshot}
          onOpenDiagnostic={(diagnostic) => {
            if (diagnostic.file)
              void window.arc.build.openDiagnostic(diagnostic.file, diagnostic.line, diagnostic.column);
          }}
          onExecute={(request: ArcBuildRequest) => {
            setLayout((current) => ({ ...current, bottomVisible: true, activeBottomPanel: 'buildOutput' }));
            void window.arc.build
              .execute(request)
              .then(setBuildSnapshot)
              .catch((reason: unknown) => {
                setLastCommand(reason instanceof Error ? reason.message : String(reason));
              });
          }}
        />
      );
    }

    if (panel === 'versionControl') return <VersionControlPanel />;

    if (panel === 'profiler') {
      return <ProfilerPanel samples={profilerSamples} />;
    }

    if (panel === 'aiAssistant') {
      return (
        <AiGatewayPanel
          status={aiGatewayStatus}
          onApprove={(requestId) => void window.arc.aiGateway.approve(requestId)}
          onDeny={(requestId) => void window.arc.aiGateway.deny(requestId)}
          onRevoke={(clientId) => void window.arc.aiGateway.revoke(clientId)}
          onCancelEdit={(sessionId, clientId) => void window.arc.aiGateway.cancelEdit(sessionId, clientId)}
          onUndoLastEdit={() => void window.arc.aiGateway.undoLastEdit()}
        />
      );
    }

    return (
      <ContentBrowserV2
        project={project}
        cache={assetCache}
        selectedAssetId={selectedAssetId}
        onSelectAsset={setSelectedAssetId}
        onCommand={runCommand}
        onInstantiatePrefab={(path) => void instantiatePrefab(path)}
        onAssetAction={async (type, guid) => {
          const response = (await window.arc.host.command(type, { guid })) as HostResponse;
          setLastCommand(response.succeeded ? `${type} completed` : response.error || `${type} failed`);
          if (response.succeeded) await refreshProjectFromHost(undefined, false);
        }}
        onTextureStreamingMode={async (guid, mode) => {
          const response = (await window.arc.host.command('asset.setTextureStreamingMode', {
            guid,
            mode,
          })) as HostResponse;
          setLastCommand(
            response.succeeded ? `Texture mode changed to ${mode}` : response.error || 'Texture mode change failed',
          );
          if (response.succeeded) await refreshProjectFromHost(undefined, false);
        }}
        thumbnailProvider={loadAssetThumbnail}
      />
    );
  };

  const renderWorkspacePanel = (panel: WorkbenchPanelId, instanceId?: string, onMaximizeToggle?: () => void) => {
    if ((dockPanelIds.center as readonly WorkbenchPanelId[]).includes(panel))
      return renderCenterPanel(panel, instanceId, onMaximizeToggle);
    if ((dockPanelIds.right as readonly WorkbenchPanelId[]).includes(panel)) return renderRightPanel(panel);
    if ((dockPanelIds.bottom as readonly WorkbenchPanelId[]).includes(panel)) return renderBottomPanel(panel);
    return renderLeftPanel(panel);
  };

  return (
    <main className="workbench-shell">
      <MenuBar
        projectTitle={`${documentState.sceneName || 'Untitled'}${documentState.dirty ? '*' : ''}`}
        canUndo={documentState.canUndo}
        canRedo={documentState.canRedo}
        undoLabel={documentState.undoLabel}
        redoLabel={documentState.redoLabel}
        onCommand={runCommand}
        gridVisible={viewportGridVisible}
        onToggleGrid={() => void toggleViewportGrid()}
        onPanel={(panel) => setRequestedWorkspacePanel(panel)}
      />
      <EditorDocumentTabs
        documents={editorDocuments}
        activeDocumentId={activeDocumentId}
        registry={editorRegistry}
        onActivate={activateDocument}
      />
      <EditorToolbarHost document={activeDocument} registry={editorRegistry} />
      <AiGatewayApprovalPrompt
        status={aiGatewayStatus}
        onApprove={(requestId) => void window.arc.aiGateway.approve(requestId)}
        onDeny={(requestId) => void window.arc.aiGateway.deny(requestId)}
        onOpenGateway={() =>
          setLayout((current) => ({
            ...current,
            bottomVisible: true,
            activeBottomPanel: 'aiAssistant',
          }))
        }
      />

      <section
        className={['workbench-body', 'workbench-body-dockview', layout.activityExpanded ? 'activity-expanded' : '']
          .filter(Boolean)
          .join(' ')}
      >
        <ActivityBar
          activeActivity={layout.activeActivity}
          expanded={layout.activityExpanded}
          onExpandedChange={setActivityExpanded}
          onSelectActivity={selectActivity}
          onSettings={() => void runCommand('settings.open')}
        />

        <section className="editor-region dockview-editor-region">
          <WorkspaceDock
            onRequestHandled={() => {
              setRequestedWorkspaceLayout(null);
              setRequestedWorkspacePanel(null);
            }}
            projectKey={project?.root ? encodeURIComponent(project.root) : 'no-project'}
            renderPanel={renderWorkspacePanel}
            requestedLayout={requestedWorkspaceLayout}
            requestedPanel={requestedWorkspacePanel}
            requestedViewportCount={viewportCount}
          />
        </section>
      </section>

      <StatusBar
        startupState={startupState}
        activeScene={project?.activeScene}
        lastCommand={lastCommand}
        aiControl={
          aiGatewayStatus?.activeEditSession
            ? `AI editing: ${aiGatewayStatus.activeEditSession.label}`
            : aiGatewayStatus?.viewportLease
              ? `AI viewport: ${aiGatewayStatus.viewportLease.clientId}`
              : undefined
        }
      />
      {settingsOpen && <SettingsDialog onClose={() => setSettingsOpen(false)} onResetLayout={resetLayout} />}
      {commandPaletteOpen && (
        <CommandPalette
          context={{ ...commandContext, modalOpen: true }}
          onClose={() => setCommandPaletteOpen(false)}
          onCommand={(command) => void runCommand(command)}
          shortcut={(command) => keybindings.current.primaryBinding(command)}
        />
      )}
      {createTerrainOpen && (
        <CreateTerrainDialog
          parent={selectedSnapshot?.entity}
          command={(type, payload) => window.arc.host.command(type, payload) as Promise<HostResponse>}
          onClose={() => setCreateTerrainOpen(false)}
          onCreated={() => void refreshProjectFromHost()}
        />
      )}
    </main>
  );
}

function PrimitivePreview({ kind }: { kind: Exclude<BasicEntityKind, 'empty' | 'terrain'> }) {
  const fillId = `primitive-fill-${kind}`;
  const common = { fill: `url(#${fillId})`, stroke: '#8fc8ff', strokeWidth: 1.35 };
  return (
    <svg className="primitive-preview" viewBox="0 0 64 64" aria-hidden="true">
      <defs>
        <linearGradient id={fillId} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#52789a" />
          <stop offset="1" stopColor="#172a3a" />
        </linearGradient>
      </defs>
      {kind === 'cube' && (
        <>
          <path {...common} d="M14 22 32 12l18 10-18 11z" />
          <path {...common} d="M14 22v22l18 10V33z" />
          <path {...common} d="M50 22v22L32 54V33z" />
        </>
      )}
      {kind === 'sphere' && (
        <>
          <circle {...common} cx="32" cy="32" r="21" />
          <ellipse cx="32" cy="32" rx="10" ry="21" fill="none" stroke="#8fc8ff" opacity="0.72" />
          <path d="M12 32h40M17 21c9 5 21 5 30 0M17 43c9-5 21-5 30 0" fill="none" stroke="#8fc8ff" opacity="0.58" />
        </>
      )}
      {kind === 'cylinder' && (
        <>
          <path {...common} d="M16 18c0-6 32-6 32 0v28c0 7-32 7-32 0z" />
          <ellipse {...common} cx="32" cy="18" rx="16" ry="6" />
          <path d="M16 45c3 7 29 7 32 0" fill="none" stroke="#8fc8ff" />
        </>
      )}
      {kind === 'cone' && (
        <>
          <path {...common} d="M32 10 13 46c1 9 37 9 38 0z" />
          <ellipse {...common} cx="32" cy="46" rx="19" ry="7" />
        </>
      )}
      {kind === 'capsule' && (
        <>
          <path {...common} d="M18 23a14 14 0 0 1 28 0v18a14 14 0 0 1-28 0z" />
          <path d="M18 23c5 4 23 4 28 0M18 41c5-4 23-4 28 0" fill="none" stroke="#8fc8ff" opacity="0.62" />
        </>
      )}
      {kind === 'plane' && (
        <>
          <path {...common} d="m8 40 29-25 19 11-29 25z" />
          <path d="m17 32 19 11M27 24l19 11M19 46l29-25" fill="none" stroke="#8fc8ff" opacity="0.55" />
        </>
      )}
    </svg>
  );
}

export function ExplorerPanel({
  project,
  selectedEntityId,
  selectedEntityIds,
  onSelectEntity,
  onRenameEntity,
  onSetEntityActive,
  onMoveEntity,
  onCreateEntity,
  onDuplicate,
  onCreatePrefab,
  onInstantiatePrefab,
  onDelete,
}: {
  project: ProjectSnapshot;
  selectedEntityId: string;
  selectedEntityIds: ReadonlySet<string>;
  onSelectEntity: (entityId: string, additive?: boolean) => void;
  onRenameEntity: (entityId: string, name: string) => void;
  onSetEntityActive: (entityId: string, active: boolean) => void;
  onMoveEntity: (entityId: string, target: SceneEntity, mode: 'before' | 'inside' | 'after') => void;
  onCreateEntity: (kind: BasicEntityKind) => void;
  onDuplicate: () => void;
  onCreatePrefab: () => void;
  onInstantiatePrefab: () => void;
  onDelete: () => void;
}) {
  const [filter, setFilter] = useState('');
  const sceneTree = useMemo(() => [sceneRootEntity(project.scene)], [project.scene]);
  const filteredScene = useMemo(() => filterSceneTree(sceneTree, filter), [sceneTree, filter]);
  const allEntities = useMemo(() => flattenScene(project.scene), [project.scene]);
  const actorCount = allEntities.length;
  const selectedCount = selectedEntityIds.size;
  const [createMenuOpen, setCreateMenuOpen] = useState(false);
  const [kindFilter, setKindFilter] = useState<'all' | SceneEntity['kind']>('all');
  const [onlyVisible, setOnlyVisible] = useState(false);
  const [savedSets, setSavedSets] = useState<Record<string, string[]>>(() => {
    try {
      return JSON.parse(localStorage.getItem('arc.hierarchy.selectionSets') ?? '{}') as Record<string, string[]>;
    } catch {
      return {};
    }
  });
  const visibleScene = useMemo(
    () =>
      filterSceneTree(filteredScene, kindFilter === 'all' ? '' : kindFilter).filter(
        (entity) => !onlyVisible || entity.visible !== false,
      ),
    [filteredScene, kindFilter, onlyVisible],
  );

  useEffect(() => {
    if (!createMenuOpen) return;
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setCreateMenuOpen(false);
    };
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('keydown', escape);
    };
  }, [createMenuOpen]);

  return (
    <div className="explorer-view">
      <Panel icon={<FolderTree size={14} />} title="Hierarchy">
        <div className="hierarchy-actions">
          <div className="hierarchy-create-menu">
            <UiIconButton
              active={createMenuOpen}
              label={createMenuOpen ? 'Close add entity drawer' : 'Add entity'}
              onClick={() => setCreateMenuOpen((open) => !open)}
            >
              <Plus size={13} />
            </UiIconButton>
          </div>
          <UiIconButton label="Duplicate selected entity" onClick={onDuplicate}>
            <Copy size={13} />
          </UiIconButton>
          <UiIconButton label="Create prefab from selection" onClick={onCreatePrefab}>
            <Box size={13} />
          </UiIconButton>
          <UiIconButton label="Instantiate prefab" onClick={onInstantiatePrefab}>
            <Database size={13} />
          </UiIconButton>
          <UiIconButton label="Delete selected entity" onClick={onDelete}>
            <Trash2 size={13} />
          </UiIconButton>
        </div>
        {createMenuOpen && (
          <section className="primitive-palette" role="menu" aria-label="Add entity">
            <div className="primitive-palette-title">
              <span>Add Entity</span>
              <UiIconButton label="Close add entity drawer" onClick={() => setCreateMenuOpen(false)}>
                <X size={12} />
              </UiIconButton>
            </div>
            <button
              className="primitive-palette-empty"
              type="button"
              role="menuitem"
              onClick={() => {
                onCreateEntity('empty');
                setCreateMenuOpen(false);
              }}
            >
              <Plus size={16} />
              Empty Entity
            </button>
            <div className="primitive-palette-heading">Basic Shapes</div>
            <div className="primitive-palette-grid">
              {(['cube', 'sphere', 'cylinder', 'cone', 'capsule', 'plane'] as const).map((kind) => (
                <button
                  key={kind}
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    onCreateEntity(kind);
                    setCreateMenuOpen(false);
                  }}
                >
                  <PrimitivePreview kind={kind} />
                  <span>{kind === 'cube' ? 'Box' : kind[0].toUpperCase() + kind.slice(1)}</span>
                </button>
              ))}
            </div>
            <div className="primitive-palette-heading">Landscape</div>
            <button
              className="primitive-palette-empty"
              type="button"
              role="menuitem"
              onClick={() => {
                onCreateEntity('terrain');
                setCreateMenuOpen(false);
              }}
            >
              <Mountain size={16} /> Terrain...
            </button>
          </section>
        )}
        <label className="hierarchy-search">
          <Search size={15} />
          <input
            aria-label="Search hierarchy"
            placeholder="Search..."
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          />
        </label>
        <div className="hierarchy-filter-bar">
          <select
            aria-label="Hierarchy kind"
            value={kindFilter}
            onChange={(event) => setKindFilter(event.target.value as typeof kindFilter)}
          >
            <option value="all">All entities</option>
            <option value="mesh">Meshes</option>
            <option value="light">Lights</option>
            <option value="camera">Cameras</option>
            <option value="environment">Environment</option>
          </select>
          <button className={onlyVisible ? 'active' : ''} onClick={() => setOnlyVisible((value) => !value)}>
            {onlyVisible ? <Eye size={12} /> : <EyeOff size={12} />} Visible
          </button>
          <details>
            <summary>Sets</summary>
            <div className="hierarchy-selection-sets">
              <button
                onClick={() => {
                  const next = { ...savedSets, Selection: [...selectedEntityIds] };
                  setSavedSets(next);
                  localStorage.setItem('arc.hierarchy.selectionSets', JSON.stringify(next));
                }}
              >
                Save Selection
              </button>
              {Object.entries(savedSets).map(([name, ids]) => (
                <button key={name} onClick={() => ids.forEach((id, index) => onSelectEntity(id, index > 0))}>
                  {name} ({ids.length})
                </button>
              ))}
            </div>
          </details>
        </div>
        <div className="hierarchy-tree">
          {visibleScene.map((entity) => (
            <SceneTreeItem
              key={entity.guid ?? entity.id}
              entity={entity}
              depth={0}
              selectedEntityId={selectedEntityId}
              selectedEntityIds={selectedEntityIds}
              onSelectEntity={onSelectEntity}
              onRenameEntity={onRenameEntity}
              onSetEntityActive={onSetEntityActive}
              onMoveEntity={onMoveEntity}
              forceExpanded={Boolean(filter)}
            />
          ))}
          {filteredScene.length === 0 && <div className="hierarchy-empty">No matching entities</div>}
        </div>
        <footer className="hierarchy-footer">
          {actorCount.toLocaleString()} actors ({selectedCount} selected)
        </footer>
      </Panel>
    </div>
  );
}

const normalizeFilterText = (value: string) =>
  value
    .toLocaleLowerCase()
    .replace(/[_./\\-]+/g, ' ')
    .trim();

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

export const filterSceneTree = (entities: SceneEntity[], filter: string): SceneEntity[] => {
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

function AssetExplorerPanel({
  project,
  selectedAssetId,
  onSelectAsset,
}: {
  project: ProjectSnapshot;
  selectedAssetId: string | null;
  onSelectAsset: (assetId: string) => void;
}) {
  return (
    <Panel icon={<Database size={14} />} title="Assets">
      <TreeSection title="Project Assets">
        {project.assets.map((asset) => (
          <AssetRow
            key={asset.id}
            asset={asset}
            selected={asset.id === selectedAssetId}
            onSelect={() => onSelectAsset(asset.id)}
          />
        ))}
      </TreeSection>
    </Panel>
  );
}

function WorldSettingsPanel({
  environment,
  assets,
  thumbnailProvider,
  onEnvironmentChange,
  onEnvironmentPreset,
  onEnvironmentHdri,
}: {
  environment: HostWorldEnvironment | null;
  assets: ReadonlyArray<AssetItem>;
  thumbnailProvider: (path: string) => Promise<string | null>;
  onEnvironmentChange: (environment: HostWorldEnvironment) => void;
  onEnvironmentPreset: (preset: string) => void;
  onEnvironmentHdri: (path: string) => Promise<boolean> | boolean | void;
}) {
  return (
    <section className="world-settings-panel">
      {environment ? (
        <WorldEnvironmentInspector
          environment={environment}
          assets={assets}
          thumbnailProvider={thumbnailProvider}
          onChange={onEnvironmentChange}
          onPreset={onEnvironmentPreset}
          onHdri={onEnvironmentHdri}
        />
      ) : (
        <PlaceholderPanel
          icon={<Settings />}
          title="World Settings"
          text="No world environment is available in this scene."
        />
      )}
    </section>
  );
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

function TreeSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="tree-section">
      <h3>
        <ChevronDown size={14} /> {title}
      </h3>
      {children}
    </section>
  );
}

function SceneTreeItem({
  entity,
  depth,
  selectedEntityId,
  selectedEntityIds,
  onSelectEntity,
  onRenameEntity,
  onSetEntityActive,
  onMoveEntity,
  forceExpanded,
}: {
  entity: SceneEntity;
  depth: number;
  selectedEntityId: string;
  selectedEntityIds: ReadonlySet<string>;
  onSelectEntity: (entityId: string, additive?: boolean) => void;
  onRenameEntity: (entityId: string, name: string) => void;
  onSetEntityActive: (entityId: string, active: boolean) => void;
  onMoveEntity: (entityId: string, target: SceneEntity, mode: 'before' | 'inside' | 'after') => void;
  forceExpanded?: boolean;
}) {
  const hasChildren = Boolean(entity.children?.length);
  const selectable = entity.id !== sceneRootId;
  const [expanded, setExpanded] = useState(true);
  const [renaming, setRenaming] = useState(false);
  const [nameDraft, setNameDraft] = useState(entity.name);
  useEffect(() => setNameDraft(entity.name), [entity.name]);
  const showChildren = hasChildren && (expanded || forceExpanded);
  return (
    <div>
      <UiTreeRow
        as="div"
        role="treeitem"
        tabIndex={0}
        className={`tree-row entity-row entity-${entity.kind}`}
        depth={depth}
        draggable={selectable}
        selected={selectable && selectedEntityIds.has(entity.id)}
        meta={
          selectable && (
            <span className="hierarchy-row-actions">
              {entity.prefabOverrideCount ? (
                <b className="prefab-override-badge" title={`${entity.prefabOverrideCount} prefab overrides`}>
                  {entity.prefabOverrideCount}
                </b>
              ) : null}
              {entity.layer && <small>{entity.layer}</small>}
              <button className="hierarchy-lock-toggle" title={entity.locked ? 'Unlock entity' : 'Lock entity'}>
                {entity.locked ? <Lock size={11} /> : <Unlock size={11} />}
              </button>
              <button
                aria-label={entity.active ? 'Disable entity' : 'Enable entity'}
                className="hierarchy-active-toggle"
                type="button"
                aria-pressed={entity.active}
                onClick={(event) => {
                  event.stopPropagation();
                  onSetEntityActive(entity.id, !entity.active);
                }}
              >
                {entity.active ? <Eye size={12} /> : <EyeOff size={12} />}
              </button>
            </span>
          )
        }
        onClick={(event) => selectable && onSelectEntity(entity.id, event.ctrlKey || event.metaKey)}
        onKeyDown={(event) => {
          if (selectable && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault();
            onSelectEntity(entity.id);
          }
        }}
        onDoubleClick={() => selectable && setRenaming(true)}
        onDragStart={(event) => event.dataTransfer.setData('application/x-arc-entity', entity.id)}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = 'move';
        }}
        onDrop={(event) => {
          event.preventDefault();
          event.stopPropagation();
          const dragged = event.dataTransfer.getData('application/x-arc-entity');
          const bounds = event.currentTarget.getBoundingClientRect();
          const ratio = (event.clientY - bounds.top) / Math.max(1, bounds.height);
          onMoveEntity(dragged, entity, ratio < 0.3 ? 'before' : ratio > 0.7 ? 'after' : 'inside');
        }}
      >
        <span
          className="hierarchy-expand"
          onClick={(event) => {
            event.stopPropagation();
            if (hasChildren) setExpanded((value) => !value);
          }}
        >
          {hasChildren && showChildren ? (
            <ChevronDown size={13} />
          ) : (
            <ChevronRight size={13} className={hasChildren ? '' : 'ghost'} />
          )}
        </span>
        <EntityIcon kind={entity.kind} />
        {renaming ? (
          <input
            autoFocus
            className="hierarchy-inline-rename"
            value={nameDraft}
            onClick={(event) => event.stopPropagation()}
            onChange={(event) => setNameDraft(event.target.value)}
            onBlur={() => {
              setRenaming(false);
              if (nameDraft.trim() !== entity.name) onRenameEntity(entity.id, nameDraft);
            }}
            onKeyDown={(event) => {
              if (event.key === 'Enter') event.currentTarget.blur();
              if (event.key === 'Escape') {
                setNameDraft(entity.name);
                setRenaming(false);
              }
            }}
          />
        ) : (
          <span>{entity.name}</span>
        )}
      </UiTreeRow>
      {showChildren &&
        entity.children?.map((child) => (
          <SceneTreeItem
            key={child.guid ?? child.id}
            entity={child}
            depth={depth + 1}
            selectedEntityId={selectedEntityId}
            selectedEntityIds={selectedEntityIds}
            onSelectEntity={onSelectEntity}
            onRenameEntity={onRenameEntity}
            onSetEntityActive={onSetEntityActive}
            onMoveEntity={onMoveEntity}
            forceExpanded={forceExpanded}
          />
        ))}
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
  return (
    <UiTreeRow className="tree-row" selected={selected} meta={<small>{asset.status}</small>} onClick={onSelect}>
      <AssetIcon kind={asset.kind} />
      <span>{asset.name}</span>
    </UiTreeRow>
  );
}

function AssetIcon({ kind }: { kind: AssetItem['kind'] }) {
  if (kind === 'scene') return <FileText size={14} />;
  if (kind === 'shader') return <FileCode2 size={14} />;
  if (kind === 'folder') return <Folder size={14} />;
  return <Database size={14} />;
}

function PlaceholderPanel({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return (
    <div className="empty-state">
      {icon}
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

void defaultWorkbenchLayout;
