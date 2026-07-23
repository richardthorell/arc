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
  Copy,
  Eye,
  EyeOff,
  MoreVertical,
  Plus,
  Lightbulb,
  Search,
  Settings,
  Trash2,
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
import { AssetThumbnail } from '../inspector/AssetPicker';
import type { AssetThumbnailProvider } from '../inspector/AssetPicker';
import type { HostEntityId, HostResponse, InspectorEntitySnapshot } from '../inspector/inspectorTypes';
import { eulerDegreesToQuaternion, hostEntityKey, parseSelectedEntitySnapshot } from '../inspector/inspectorTypes';
import { ProfilerPanel } from '../profiler/ProfilerPanel';
import type { ProfilerSnapshot } from '../profiler/ProfilerPanel';

import './workbench.css';

export type HostSceneEntity = {
  entity: HostEntityId;
  guid: string;
  parentGuid: string;
  siblingOrder: number;
  name: string;
  kind: 'camera' | 'light' | 'environment' | 'mesh' | 'primitive' | 'imported' | 'unknown';
  active: boolean;
  selected: boolean;
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

type SceneDocumentState = Omit<HostSceneSnapshot, 'entities'>;

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

export const buildSceneTree = (entities: HostSceneEntity[]): SceneEntity[] => {
  const byGuid = new Map<string, SceneEntity>();
  for (const entity of entities) {
    byGuid.set(entity.guid, {
      id: hostEntityKey(entity.entity),
      guid: entity.guid,
      name: entity.name,
      kind: sceneKindFromHost(entity.kind),
      active: entity.active,
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
    return nextSelection === selectedEntityId ? 'none' : 'selection';
  }
  if (event.type === 'component.changed') {
    if (!validHostEntity(event.entity)) return 'none';
    return hostEntityKey(event.entity) === selectedEntityId ? 'selected' : 'hierarchy';
  }
  if (event.type === 'scene.changed' || event.type === 'entity.created' || event.type === 'entity.deleted' ||
      event.type === 'project.opened' || event.type === 'project.closed') return 'all';
  return 'none';
};

const resizeLimits = {
  left: { min: 220, max: 520 },
  right: { min: 300, max: 640 },
  bottom: { min: 112, max: 420 },
};
const translationSnapOptions = [0.05, 0.1, 0.25, 0.5, 1] as const;
const rotationSnapOptions = [5, 10, 15, 30, 45, 90] as const;
const scaleSnapOptions = [0.05, 0.1, 0.25, 0.5] as const;
const nextSnapOption = (options: readonly number[], current: number) => options[(options.indexOf(current) + 1) % options.length];

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
    meshRenderer: entity.kind === 'mesh' ? {
      visible: true,
      baseColorTint: { x: 1, y: 1, z: 1, w: 1 },
      hasMaterial: true,
      assetBackedMaterial: true,
      materialName: 'Default Material',
      materialPath: 'materials/default.arcmat',
    } : null,
    terrain: null,
    components: [
      { kind: 'transform', label: 'Transform', editable: true },
      ...(entity.kind === 'camera' ? [{ kind: 'camera', label: 'Camera', editable: true }] : []),
      ...(entity.kind === 'mesh' ? [{ kind: 'meshRenderer', label: 'Mesh Renderer', editable: true }] : []),
    ],
  };
};

export function Workbench() {
  const { layout, setLayout, resetLayout } = useWorkbenchLayout();
  const [startupState, setStartupState] = useState<StartupState | null>(null);
  const [project, setProject] = useState<ProjectSnapshot | null>(null);
  const [hostConsoleEvents, setHostConsoleEvents] = useState<ConsoleEvent[]>([]);
  const [selectedEntityId, setSelectedEntityId] = useState('camera-main');
  const selectedEntityIdRef = useRef(selectedEntityId);
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>('asset-scene-demo');
  const [selectedSnapshot, setSelectedSnapshot] = useState<InspectorEntitySnapshot | null>(null);
  const [selectedSnapshotLoading, setSelectedSnapshotLoading] = useState(false);
  const selectedSnapshotRevision = useRef(0);
  const hostEventRefreshTimer = useRef<number | null>(null);
  const hostEventRefreshMode = useRef<'none' | 'selected' | 'hierarchy' | 'all'>('none');
  const [worldEnvironment, setWorldEnvironment] = useState<HostWorldEnvironment | null>(null);
  const [documentState, setDocumentState] = useState<SceneDocumentState>({
    sceneGuid: '', sceneName: 'Untitled', activeScenePath: '', dirty: true,
    canUndo: false, canRedo: false, undoLabel: '', redoLabel: '',
  });
  const [activeTool, setActiveTool] = useState<'select' | 'translate' | 'rotate' | 'scale' | 'terrain'>('translate');
  const [coordinateSpace, setCoordinateSpace] = useState<'world' | 'local'>('world');
  const [snapping, setSnapping] = useState(false);
  const [translationSnap, setTranslationSnap] = useState(0.25);
  const [rotationSnap, setRotationSnap] = useState(15);
  const [scaleSnap, setScaleSnap] = useState(0.1);
  const [lastCommand, setLastCommand] = useState('Workbench ready');
  const [profilerSamples, setProfilerSamples] = useState<ProfilerSnapshot[]>([]);

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

  useEffect(() => window.arc?.host?.onEvent?.((event) => {
    if (event.type === 'profiler.snapshot' && event.payload && typeof event.payload === 'object') {
      const sample = event.payload as ProfilerSnapshot;
      setProfilerSamples((current) => [...current, sample].slice(-3000));
      return;
    }
    setLastCommand(event.message || event.type);
    if (event.payload && typeof event.payload === 'object' && 'tool' in event.payload) {
      const payload = event.payload as { tool?: unknown; coordinateSpace?: unknown; snapping?: unknown;
        translationSnap?: unknown; rotationSnapDegrees?: unknown; scaleSnap?: unknown };
      const tool = String(payload.tool);
      if (tool === 'select' || tool === 'translate' || tool === 'rotate' || tool === 'scale') setActiveTool(tool);
      if (payload.coordinateSpace === 'world' || payload.coordinateSpace === 'local') setCoordinateSpace(payload.coordinateSpace);
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
  }), []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.matches('input, textarea, select, [contenteditable="true"]')) return;
      const command = event.ctrlKey && event.shiftKey && event.key.toLocaleLowerCase() === 's' ? 'file.saveAs'
        : event.ctrlKey && event.key.toLocaleLowerCase() === 's' ? 'file.save'
          : event.ctrlKey && event.key.toLocaleLowerCase() === 'z' ? 'edit.undo'
            : event.ctrlKey && (event.key.toLocaleLowerCase() === 'y') ? 'edit.redo'
              : event.ctrlKey && event.key.toLocaleLowerCase() === 'd' ? 'entity.duplicate'
                : event.key === 'Delete' ? 'entity.delete'
                  : event.key.toLocaleLowerCase() === 'q' ? 'viewport.select'
                    : event.key.toLocaleLowerCase() === 'w' ? 'viewport.translate'
                      : event.key.toLocaleLowerCase() === 'e' ? 'viewport.rotate'
                        : event.key.toLocaleLowerCase() === 'r' ? 'viewport.scale'
                          : event.key.toLocaleLowerCase() === 'f' ? 'viewport.frameSelected' : null;
      if (!command) return;
      event.preventDefault();
      void runCommand(command);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  });

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

    if (startupState?.engineHostConnected && window.arc?.host) {
      try {
        let response: HostResponse | undefined;
        if (command === 'file.new') {
          response = await window.arc.host.command('scene.new', { name: 'Untitled' }) as HostResponse;
        } else if (command === 'file.save' || command === 'assets.saveAll') {
          if (!documentState.activeScenePath) {
            const result = await window.arc.dialog.saveScene();
            if (result.canceled) return setLastCommand('Save canceled');
            response = result.response as HostResponse;
          } else {
            response = await window.arc.host.command('scene.save') as HostResponse;
          }
        } else if (command === 'file.saveAs') {
          const result = await window.arc.dialog.saveScene();
          if (result.canceled) return setLastCommand('Save canceled');
          response = result.response as HostResponse;
        } else if (command === 'edit.undo') {
          response = await window.arc.host.command('history.undo') as HostResponse;
        } else if (command === 'edit.redo') {
          response = await window.arc.host.command('history.redo') as HostResponse;
        } else if (command === 'entity.duplicate' && selectedSnapshot) {
          response = await window.arc.host.command('entity.duplicate', { entity: selectedSnapshot.entity }) as HostResponse;
        } else if (command === 'entity.delete' && selectedSnapshot) {
          response = await window.arc.host.command('entity.delete', { entity: selectedSnapshot.entity }) as HostResponse;
        } else if (command.startsWith('viewport.')) {
          if (command === 'viewport.frameSelected') {
            await window.arc.viewport.cameraInput({ focusSelected: true });
            return setLastCommand('Framed selected entity');
          }
          const tool = command.slice('viewport.'.length) as typeof activeTool;
          if (tool === 'select' || tool === 'translate' || tool === 'rotate' || tool === 'scale' || tool === 'terrain') {
            response = await window.arc.host.command('viewport.setTool', {
              tool, coordinateSpace, snapping,
              translationSnap, rotationSnapDegrees: rotationSnap, scaleSnap,
            }) as HostResponse;
            if (response?.succeeded) setActiveTool(tool);
          }
        }
        if (response) {
          setLastCommand(response.succeeded ? `${command} completed` : response.error || `${command} failed`);
          const responsePath = response.payload && typeof response.payload === 'object' && 'path' in response.payload
            ? String((response.payload as { path?: unknown }).path ?? '') : undefined;
          if (response.succeeded) await refreshProjectFromHost(responsePath);
          return;
        }
      } catch (error) {
        setLastCommand(error instanceof Error ? error.message : String(error));
        return;
      }
    }

    const result = await executeWorkbenchCommand(command);
    setLastCommand(result.message);
  };

  const refreshProjectFromHost = async (activeScene?: string, refreshSelection = true) => {
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

    const scenePayload = sceneResponse.payload;
    const hostEntities = scenePayload.entities.filter((entity) => !isEditorOnlyHostEntity(entity));
    const scene = buildSceneTree(hostEntities);
    const { entities: _entities, ...nextDocumentState } = scenePayload;
    setDocumentState(nextDocumentState);

    const hostAssets = assetsResponse.succeeded && assetsResponse.payload ? assetsResponse.payload : null;
    const assets = hostAssets?.assets.map((asset): AssetItem => ({
      id: asset.path,
      name: assetNameFromPath(asset.path),
      path: asset.path,
      kind: assetKindFromHost(asset.kind),
      status: asset.importRunning ? 'importing' : asset.imported ? 'ready' : 'missing',
    })) ?? project?.assets ?? [];

    const selected = hostEntities.find((entity) => entity.selected);
    if (selected && refreshSelection) {
      const selectedKey = hostEntityKey(selected.entity);
      selectedEntityIdRef.current = selectedKey;
      setSelectedEntityId(selectedKey);
      await refreshSelectedEntity(selectedKey, true);
    } else if (refreshSelection) {
      selectedEntityIdRef.current = '';
      setSelectedEntityId('');
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
      activeScene: activeScene ?? scenePayload.activeScenePath ?? current?.activeScene ?? '',
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
    if (result.selectedEntityId === selectedEntityIdRef.current) return;
    selectedEntityIdRef.current = result.selectedEntityId;
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

  const mutateHierarchyEntity = async (type: string, payload: Record<string, unknown>) => {
    if (!startupState?.engineHostConnected) return false;
    const response = await window.arc.host.command(type, payload) as HostResponse;
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

  const createHierarchyChild = () => {
    const parent = selectedSnapshot?.entity;
    void mutateHierarchyEntity('entity.create', { kind: 'empty', ...(parent ? { parent } : {}) });
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
      ? allEntities.find((value) => value.id === target.parentId)?.children ?? []
      : project?.scene ?? [];
    const targetIndex = siblings.findIndex((value) => value.id === target.id);
    const beforeTarget = mode === 'before' ? target : siblings[targetIndex + 1];
    const beforeSibling = beforeTarget ? parseHostEntityId(beforeTarget.id) : null;
    if (source?.parentId === target.parentId) {
      void mutateHierarchyEntity('entity.reorder', { entity, ...(beforeSibling ? { beforeSibling } : {}) });
      return;
    }
    const parent = target.parentId ? parseHostEntityId(target.parentId) : null;
    void mutateHierarchyEntity('entity.reparent', {
      entity, ...(parent ? { parent } : {}), ...(beforeSibling ? { beforeSibling } : {}), preserveWorld: true,
    });
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
  const updateViewportToolOptions = async (nextSpace: 'world' | 'local', nextSnapping: boolean,
    nextTranslationSnap = translationSnap, nextRotationSnap = rotationSnap, nextScaleSnap = scaleSnap) => {
    if (startupState?.engineHostConnected) {
      const response = await window.arc.host.command('viewport.setTool', {
        tool: activeTool, coordinateSpace: nextSpace, snapping: nextSnapping,
        translationSnap: nextTranslationSnap, rotationSnapDegrees: nextRotationSnap, scaleSnap: nextScaleSnap,
      }) as HostResponse;
      if (!response.succeeded) return setLastCommand(response.error || 'Viewport tool update failed');
    }
    setCoordinateSpace(nextSpace);
    setSnapping(nextSnapping);
    setTranslationSnap(nextTranslationSnap);
    setRotationSnap(nextRotationSnap);
    setScaleSnap(nextScaleSnap);
  };

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
      return <ExplorerPanel project={project} selectedEntityId={selectedEntityId} onSelectEntity={selectEntity}
        onRenameEntity={renameHierarchyEntity} onSetEntityActive={setHierarchyEntityActive} onMoveEntity={moveHierarchyEntity}
        onCreateChild={createHierarchyChild} onDuplicate={() => void runCommand('entity.duplicate')}
        onDelete={() => void runCommand('entity.delete')} />;
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
        command={async (type, payload, edit) => {
          if (!startupState?.engineHostConnected) return { succeeded: true };
          return window.arc.host.command(type, payload, edit) as Promise<HostResponse>;
        }}
        loading={selectedSnapshotLoading}
        snapshot={selectedSnapshot}
        assets={project?.assets ?? []}
        thumbnailProvider={loadAssetThumbnail}
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
      return <ContentBrowserPanel project={project} selectedAssetId={selectedAssetId} onSelectAsset={setSelectedAssetId}
        onCommand={runCommand} thumbnailProvider={loadAssetThumbnail} />;
    }

    if (panel === 'console') {
      return <ConsolePanel events={[...(project?.console ?? []), ...hostConsoleEvents]} lastCommand={lastCommand} />;
    }

    if (panel === 'profiler') {
      return <ProfilerPanel samples={profilerSamples} />;
    }

    return <ContentBrowserPanel project={project} selectedAssetId={selectedAssetId} onSelectAsset={setSelectedAssetId}
      onCommand={runCommand} thumbnailProvider={loadAssetThumbnail} />;
  };

  const workbenchBodyStyle = {
    '--arc-left-panel-width': `${layout.leftPanelWidth}px`,
    '--arc-right-panel-width': `${layout.rightPanelWidth}px`,
    '--arc-bottom-panel-height': `${layout.bottomPanelHeight}px`,
  } as CSSProperties;

  return (
    <main className="workbench-shell">
      <MenuBar projectTitle={`${documentState.sceneName || 'Untitled'}${documentState.dirty ? '*' : ''}`}
        canUndo={documentState.canUndo} canRedo={documentState.canRedo} undoLabel={documentState.undoLabel}
        redoLabel={documentState.redoLabel} onCommand={runCommand} />
      <MainToolbar activeTool={activeTool} coordinateSpace={coordinateSpace} snapping={snapping}
        terrainEnabled={selectedSnapshot?.terrain !== null && selectedSnapshot?.terrain !== undefined}
        translationSnap={translationSnap} rotationSnap={rotationSnap} scaleSnap={scaleSnap} onCommand={runCommand}
        onToggleCoordinateSpace={() => void updateViewportToolOptions(coordinateSpace === 'world' ? 'local' : 'world', snapping)}
        onToggleSnapping={() => void updateViewportToolOptions(coordinateSpace, !snapping)}
        onCycleTranslationSnap={() => void updateViewportToolOptions(coordinateSpace, snapping,
          nextSnapOption(translationSnapOptions, translationSnap))}
        onCycleRotationSnap={() => void updateViewportToolOptions(coordinateSpace, snapping, translationSnap,
          nextSnapOption(rotationSnapOptions, rotationSnap))}
        onCycleScaleSnap={() => void updateViewportToolOptions(coordinateSpace, snapping, translationSnap, rotationSnap,
          nextSnapOption(scaleSnapOptions, scaleSnap))} />

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

function ExplorerPanel({ project, selectedEntityId, onSelectEntity, onRenameEntity, onSetEntityActive, onMoveEntity, onCreateChild, onDuplicate, onDelete }: {
  project: ProjectSnapshot;
  selectedEntityId: string;
  onSelectEntity: (entityId: string) => void;
  onRenameEntity: (entityId: string, name: string) => void;
  onSetEntityActive: (entityId: string, active: boolean) => void;
  onMoveEntity: (entityId: string, target: SceneEntity, mode: 'before' | 'inside' | 'after') => void;
  onCreateChild: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
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
        <div className="hierarchy-actions">
          <UiIconButton label="Create child entity" onClick={onCreateChild}><Plus size={13} /></UiIconButton>
          <UiIconButton label="Duplicate selected entity" onClick={onDuplicate}><Copy size={13} /></UiIconButton>
          <UiIconButton label="Delete selected entity" onClick={onDelete}><Trash2 size={13} /></UiIconButton>
        </div>
        <label className="hierarchy-search">
          <Search size={15} />
          <input aria-label="Search hierarchy" placeholder="Search..." value={filter} onChange={(event) => setFilter(event.target.value)} />
        </label>
        <div className="hierarchy-tree">
          {filteredScene.map((entity) => (
            <SceneTreeItem key={entity.guid ?? entity.id} entity={entity} depth={0} selectedEntityId={selectedEntityId}
              onSelectEntity={onSelectEntity} onRenameEntity={onRenameEntity} onSetEntityActive={onSetEntityActive}
              onMoveEntity={onMoveEntity} forceExpanded={Boolean(filter)} />
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

function ContentBrowserPanel({ project, selectedAssetId, onSelectAsset, onCommand, thumbnailProvider }: {
  project: ProjectSnapshot | null;
  selectedAssetId: string | null;
  onSelectAsset: (assetId: string) => void;
  onCommand: (command: CommandId) => void;
  thumbnailProvider: AssetThumbnailProvider;
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
        {(project?.assets ?? []).map((asset) => <AssetCard key={asset.id} asset={asset} selected={asset.id === selectedAssetId}
          thumbnailProvider={thumbnailProvider} onSelect={() => onSelectAsset(asset.id)} />)}
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

function SceneTreeItem({ entity, depth, selectedEntityId, onSelectEntity, onRenameEntity, onSetEntityActive, onMoveEntity, forceExpanded }: {
  entity: SceneEntity;
  depth: number;
  selectedEntityId: string;
  onSelectEntity: (entityId: string) => void;
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
        selected={selectable && entity.id === selectedEntityId}
        meta={selectable && <button aria-label={entity.active ? 'Disable entity' : 'Enable entity'} className="hierarchy-active-toggle" type="button"
          aria-pressed={entity.active} onClick={(event) => { event.stopPropagation(); onSetEntityActive(entity.id, !entity.active); }}>
          {entity.active ? <Eye size={12} /> : <EyeOff size={12} />}
        </button>}
        onClick={() => selectable && onSelectEntity(entity.id)}
        onKeyDown={(event) => {
          if (selectable && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault();
            onSelectEntity(entity.id);
          }
        }}
        onDoubleClick={() => selectable && setRenaming(true)}
        onDragStart={(event) => event.dataTransfer.setData('application/x-arc-entity', entity.id)}
        onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'move'; }}
        onDrop={(event) => {
          event.preventDefault();
          event.stopPropagation();
          const dragged = event.dataTransfer.getData('application/x-arc-entity');
          const bounds = event.currentTarget.getBoundingClientRect();
          const ratio = (event.clientY - bounds.top) / Math.max(1, bounds.height);
          onMoveEntity(dragged, entity, ratio < 0.3 ? 'before' : ratio > 0.7 ? 'after' : 'inside');
        }}
      >
        <span className="hierarchy-expand" onClick={(event) => { event.stopPropagation(); if (hasChildren) setExpanded((value) => !value); }}>
          {hasChildren && showChildren ? <ChevronDown size={13} /> : <ChevronRight size={13} className={hasChildren ? '' : 'ghost'} />}
        </span>
        <EntityIcon kind={entity.kind} />
        {renaming ? <input autoFocus className="hierarchy-inline-rename" value={nameDraft}
          onClick={(event) => event.stopPropagation()} onChange={(event) => setNameDraft(event.target.value)}
          onBlur={() => { setRenaming(false); if (nameDraft.trim() !== entity.name) onRenameEntity(entity.id, nameDraft); }}
          onKeyDown={(event) => {
            if (event.key === 'Enter') event.currentTarget.blur();
            if (event.key === 'Escape') { setNameDraft(entity.name); setRenaming(false); }
          }} /> : <span>{entity.name}</span>}
      </UiTreeRow>
      {showChildren && entity.children?.map((child) => <SceneTreeItem key={child.guid ?? child.id} entity={child} depth={depth + 1}
        selectedEntityId={selectedEntityId} onSelectEntity={onSelectEntity} onRenameEntity={onRenameEntity}
        onSetEntityActive={onSetEntityActive} onMoveEntity={onMoveEntity} forceExpanded={forceExpanded} />)}
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

function AssetCard({ asset, selected, thumbnailProvider, onSelect }: {
  asset: AssetItem;
  selected: boolean;
  thumbnailProvider: AssetThumbnailProvider;
  onSelect: () => void;
}) {
  const draggable = asset.kind === 'texture' || asset.kind === 'material';
  return <UiButton className={selected ? 'asset-card-foundation selected' : 'asset-card-foundation'} draggable={draggable} onDragStart={(event) => {
    if (!draggable) return;
    event.dataTransfer.setData('application/x-arc-asset', asset.path);
    if (asset.kind === 'texture') event.dataTransfer.setData('application/x-arc-environment', asset.path);
  }} onClick={onSelect} variant="default">
    {draggable ? <AssetThumbnail asset={asset} path={asset.path} provider={thumbnailProvider} /> : <AssetIcon kind={asset.kind} />}
    <strong>{asset.name}</strong><span>{asset.kind}</span>
  </UiButton>;
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
