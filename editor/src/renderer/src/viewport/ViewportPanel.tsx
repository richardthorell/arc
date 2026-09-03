import { useCallback, useEffect, useRef, useState } from 'react';
import type { DragEvent, KeyboardEvent, PointerEvent, WheelEvent } from 'react';
import { Box, Camera, Eye, EyeOff, Focus, Maximize2, RefreshCw } from 'lucide-react';

import type { CommandId } from '../app/workbenchTypes';
import type { StartupState } from '../app/workbenchTypes';
import type { ProjectSnapshot } from '../services/editorHostTypes';
import { arcAssetDragMime, arcEnvironmentDragMime, readArcAssetDragPayload } from '../services/assetDragPayload';
import { assignDroppedMaterialToViewport, instantiateDroppedMeshInViewport } from './viewportAssetDrop';

import './viewport.css';
import { toViewportPixels } from './viewportCoordinates';
import { viewportFlyMovement, viewportFlyMovementCodes } from './viewportFlyNavigation';
import { normalizeViewportWheel } from './viewportWheel';

type ViewportPanelProps = {
  viewportId?: string;
  project: ProjectSnapshot | null;
  startupState: StartupState | null;
  onCommand: (command: CommandId) => void;
  onReconnect: () => Promise<void>;
  gridVisible?: boolean;
  onGridVisibilityChange?: (visible: boolean) => void;
  onFocusChange?: (focused: boolean) => void;
  onMaximizeToggle?: () => void;
  onViewportLayoutChange?: (count: 1 | 2 | 3 | 4) => void;
  active?: boolean;
};

type DragState = {
  pointerId: number;
  mode: 'orbit' | 'pan' | 'look';
  x: number;
  y: number;
};

type HostResponse<T> = {
  kind: 'response';
  requestId: number;
  succeeded: boolean;
  error: string;
  payload: T;
};

type ViewportStats = {
  viewportId?: string;
  width: number;
  height: number;
  fps: number;
  frameTimeMs: number;
  drawCalls: number;
  frameIndex: number;
  submitted: boolean;
  renderOptions?: ViewportRenderOptions;
  camera?: {
    transform: { position: [number, number, number] };
    focus: { position: [number, number, number] };
  } | null;
};

type ViewportRenderOptions = {
  renderMode: 'shaded' | 'wireframe';
  visualization: string;
  overlay: 'none' | 'selectedWireframe' | 'allWireframe';
  shadows: boolean;
  grid: boolean;
  realtime: boolean;
  cameraSpeed: number;
  antiAliasing: 'inherit' | 'disabled' | 'fxaa' | 'taa' | 'taau';
  environment: {
    sky: boolean;
    fog: boolean;
    terrain: boolean;
    water: boolean;
    vegetation: boolean;
    decals: boolean;
  };
};

const defaultRenderOptions: ViewportRenderOptions = {
  renderMode: 'shaded',
  visualization: 'standard',
  overlay: 'selectedWireframe',
  shadows: true,
  grid: true,
  realtime: true,
  cameraSpeed: 4,
  antiAliasing: 'inherit',
  environment: { sky: true, fog: true, terrain: true, water: true, vegetation: true, decals: true },
};

type ViewportCommandResponse = {
  succeeded?: boolean;
  error?: string;
};

type ViewportBounds = {
  viewportId: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

const boundsKey = (bounds: ViewportBounds) =>
  `${bounds.viewportId}:${bounds.x}:${bounds.y}:${bounds.width}:${bounds.height}`;

const fallbackStats = (project: ProjectSnapshot | null): ViewportStats => ({
  width: 0,
  height: 0,
  fps: project?.renderStats.fps ?? 0,
  frameTimeMs: project?.renderStats.frameTimeMs ?? 0,
  drawCalls: project?.renderStats.drawCalls ?? 0,
  frameIndex: 0,
  submitted: false,
});

const formatNumber = (value: number) => Math.max(0, value).toLocaleString();
const formatFps = (value: number) => (Number.isFinite(value) && value > 0 ? value.toFixed(0) : '--');
const formatFrameTime = (value: number) => (Number.isFinite(value) && value > 0 ? value.toFixed(2) : '--');

export function ViewportPanel({
  viewportId = 'viewport-1',
  project,
  startupState,
  onCommand,
  onReconnect,
  gridVisible: controlledGridVisible,
  onGridVisibilityChange,
  onFocusChange,
  onMaximizeToggle,
  onViewportLayoutChange,
  active = true,
}: ViewportPanelProps) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const surfaceId = `arc-viewport-surface-${viewportId.replaceAll(/[^a-zA-Z0-9_-]/g, '-')}`;
  const dragRef = useRef<DragState | null>(null);
  const flyNavigationActiveRef = useRef(false);
  const movementKeysRef = useRef(new Set<string>());
  const consumedMovementKeysRef = useRef(new Set<string>());
  const movementLastTickRef = useRef<number | null>(null);
  const lastAttachAttemptRef = useRef(0);
  const viewportAttachedRef = useRef(false);
  const lastViewportBoundsRef = useRef('');
  const pendingViewportBoundsRef = useRef<ViewportBounds | null>(null);
  const resizeInFlightRef = useRef(false);
  const sharedFailureRef = useRef('');
  const [viewportError, setViewportError] = useState('');
  const [sharedFailure, setSharedFailure] = useState('');
  const [viewportStats, setViewportStats] = useState<ViewportStats>(() => fallbackStats(project));
  const [localGridVisible, setLocalGridVisible] = useState(true);
  const [projection, setProjection] = useState('perspective');
  const [transport, setTransport] = useState<'unavailable' | 'native' | 'streamed'>(
    startupState?.viewportMode ?? 'unavailable',
  );
  const gridVisible = controlledGridVisible ?? localGridVisible;
  const viewportAvailable = (transport === 'native' || transport === 'streamed') && Boolean(window.arc?.viewport);
  const streamedAvailable = transport === 'streamed' && Boolean(window.arc?.viewport);
  const viewportActive = active && viewportAvailable;
  const stats = viewportActive ? viewportStats : fallbackStats(project);

  const viewportBounds = useCallback(() => {
    const element = bodyRef.current;
    if (!element) {
      return null;
    }
    const rect = element.getBoundingClientRect();
    return {
      viewportId,
      x: Math.round(rect.left),
      y: Math.round(rect.top),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
    };
  }, [viewportId]);

  useEffect(() => {
    const nextTransport = startupState?.viewportMode ?? 'unavailable';
    setTransport(nextTransport);
    sharedFailureRef.current = '';
    setSharedFailure('');
    viewportAttachedRef.current = false;
    setViewportError('');
  }, [startupState]);

  const recordSharedFailure = useCallback((reason: string) => {
    const detail = reason.trim() || 'Unknown shared-texture error';
    sharedFailureRef.current = detail;
    setSharedFailure(detail);
    viewportAttachedRef.current = false;
    setViewportError(`Shared GPU viewport failed: ${detail}`);
  }, []);

  const attachViewport = useCallback(async () => {
    if (!viewportActive || (streamedAvailable && sharedFailureRef.current)) {
      return;
    }
    const bounds = viewportBounds();
    if (!bounds || bounds.width < 2 || bounds.height < 2) {
      return;
    }
    try {
      lastAttachAttemptRef.current = Date.now();
      const response = (await (streamedAvailable
        ? window.arc.viewport.create(bounds)
        : window.arc.viewport.attach(bounds))) as ViewportCommandResponse | undefined;
      if (response?.succeeded === false) {
        if (streamedAvailable) {
          recordSharedFailure(response.error || 'Unknown shared-texture error');
          return;
        }
        throw new Error(response.error || 'Viewport attachment was rejected');
      }
      viewportAttachedRef.current = true;
      lastViewportBoundsRef.current = boundsKey(bounds);
      if (streamedAvailable) {
        sharedFailureRef.current = '';
        setSharedFailure('');
      }
      setViewportError('');
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      if (streamedAvailable) {
        recordSharedFailure(reason);
        return;
      }
      setViewportError(reason);
    }
  }, [recordSharedFailure, streamedAvailable, viewportActive, viewportBounds]);

  const retrySharedViewport = useCallback(() => {
    if (!streamedAvailable) return;
    sharedFailureRef.current = '';
    setSharedFailure('');
    setViewportError('');
    lastAttachAttemptRef.current = 0;
    void attachViewport();
  }, [attachViewport, streamedAvailable]);

  const activateNativeFallback = useCallback(async () => {
    if (!streamedAvailable || !sharedFailureRef.current) return;
    const bounds = viewportBounds();
    if (!bounds || bounds.width < 2 || bounds.height < 2) return;
    try {
      lastAttachAttemptRef.current = Date.now();
      const response = (await window.arc.viewport.attach(bounds)) as ViewportCommandResponse | undefined;
      if (response?.succeeded === false) throw new Error(response.error || 'Native viewport fallback was rejected');
      viewportAttachedRef.current = true;
      lastViewportBoundsRef.current = boundsKey(bounds);
      setTransport('native');
      setViewportError('');
    } catch (error) {
      setViewportError(error instanceof Error ? error.message : String(error));
    }
  }, [streamedAvailable, viewportBounds]);

  const resizeViewport = useCallback(() => {
    if (!viewportActive || !viewportAttachedRef.current) {
      return;
    }
    const bounds = viewportBounds();
    if (!bounds || bounds.width < 2 || bounds.height < 2) {
      return;
    }
    const key = boundsKey(bounds);
    if (key === lastViewportBoundsRef.current) return;
    lastViewportBoundsRef.current = key;
    pendingViewportBoundsRef.current = bounds;
    if (resizeInFlightRef.current) return;

    resizeInFlightRef.current = true;
    void (async () => {
      try {
        while (pendingViewportBoundsRef.current) {
          const next = pendingViewportBoundsRef.current;
          pendingViewportBoundsRef.current = null;
          const response = (await window.arc.viewport.resize(next)) as ViewportCommandResponse | undefined;
          if (response?.succeeded === false) throw new Error(response.error || 'Native viewport resize was rejected');
        }
        setViewportError('');
      } catch (error) {
        lastViewportBoundsRef.current = '';
        setViewportError(error instanceof Error ? error.message : String(error));
      } finally {
        resizeInFlightRef.current = false;
      }
    })();
  }, [viewportActive, viewportBounds]);

  useEffect(() => {
    void attachViewport();
    return () => {
      viewportAttachedRef.current = false;
      pendingViewportBoundsRef.current = null;
      lastViewportBoundsRef.current = '';
      if (viewportActive) void window.arc.viewport.detach?.(viewportId);
    };
  }, [attachViewport, viewportActive, viewportId]);

  useEffect(() => {
    if (!streamedAvailable) return;
    window.arc.viewport.registerSurface?.(viewportId, surfaceId);
    return () => window.arc.viewport.unregisterSurface?.(viewportId);
  }, [streamedAvailable, surfaceId, viewportId]);

  useEffect(() => {
    if (!viewportAvailable) return;
    void window.arc.viewport.setVisibility?.(viewportId, active);
  }, [active, viewportAvailable, viewportId]);

  useEffect(() => {
    const element = bodyRef.current;
    if (!element || !viewportActive) {
      return;
    }

    const observer = new ResizeObserver(resizeViewport);
    observer.observe(element);
    let animationFrame = 0;
    const trackBounds = () => {
      resizeViewport();
      animationFrame = window.requestAnimationFrame(trackBounds);
    };
    animationFrame = window.requestAnimationFrame(trackBounds);
    window.addEventListener('resize', resizeViewport);
    return () => {
      window.cancelAnimationFrame(animationFrame);
      observer.disconnect();
      window.removeEventListener('resize', resizeViewport);
    };
  }, [viewportActive, resizeViewport]);

  useEffect(() => {
    if (!viewportActive || !window.arc?.host) {
      setViewportStats(fallbackStats(project));
      return;
    }

    let cancelled = false;
    const pollStats = async () => {
      try {
        const response = (await window.arc.host.query('viewport.state', { viewportId })) as HostResponse<ViewportStats>;
        if (!cancelled && response?.succeeded && response.payload) {
          setViewportStats(response.payload);
          if (typeof response.payload.renderOptions?.grid === 'boolean')
            setLocalGridVisible(response.payload.renderOptions.grid);
          if (!streamedAvailable || !sharedFailureRef.current) setViewportError('');
          if (
            (!response.payload.submitted || response.payload.frameIndex === 0) &&
            (!streamedAvailable || !sharedFailureRef.current) &&
            Date.now() - lastAttachAttemptRef.current >= 1000
          )
            await attachViewport();
        }
      } catch (error) {
        if (!cancelled) {
          setViewportError(error instanceof Error ? error.message : String(error));
        }
      }
    };

    void pollStats();
    const interval = window.setInterval(pollStats, 500);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [attachViewport, project, streamedAvailable, viewportActive, viewportId]);

  const sendCameraInput = (input: Parameters<typeof window.arc.viewport.cameraInput>[0]) => {
    void window.arc.viewport.cameraInput({ ...input, viewportId }).catch((error) => {
      setViewportError(error instanceof Error ? error.message : String(error));
    });
  };

  useEffect(() => {
    if (!viewportActive) {
      flyNavigationActiveRef.current = false;
      movementKeysRef.current.clear();
      consumedMovementKeysRef.current.clear();
      movementLastTickRef.current = null;
      return;
    }

    const timer = window.setInterval(() => {
      const now = performance.now();
      if (!flyNavigationActiveRef.current || movementKeysRef.current.size === 0) {
        movementLastTickRef.current = now;
        return;
      }
      const previous = movementLastTickRef.current ?? now;
      movementLastTickRef.current = now;
      const deltaSeconds = Math.min(Math.max((now - previous) / 1000, 0), 0.05);
      const movement = viewportFlyMovement(movementKeysRef.current, deltaSeconds * 4);
      if (!movement) return;
      void window.arc.viewport.cameraInput({ ...movement, viewportId }).catch((error) => {
        setViewportError(error instanceof Error ? error.message : String(error));
      });
    }, 16);

    return () => window.clearInterval(timer);
  }, [viewportActive, viewportId]);

  const pointerCoordinates = (clientX: number, clientY: number) => {
    const rect = bodyRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return toViewportPixels({
      clientX,
      clientY,
      left: rect.left,
      top: rect.top,
      cssWidth: rect.width,
      cssHeight: rect.height,
      renderWidth: stats.width,
      renderHeight: stats.height,
      devicePixelRatio: window.devicePixelRatio,
    });
  };

  const onAssetDragOver = (event: DragEvent<HTMLDivElement>) => {
    if (!viewportActive) return;
    const externalModel =
      event.dataTransfer.types.includes('Files') &&
      Array.from(event.dataTransfer.files).some((file) => /\.(fbx|glb|gltf|obj)$/i.test(file.name));
    if (
      !externalModel &&
      !event.dataTransfer.types.includes(arcAssetDragMime) &&
      !event.dataTransfer.types.includes(arcEnvironmentDragMime)
    )
      return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  };

  const onAssetDrop = (event: DragEvent<HTMLDivElement>) => {
    if (!viewportActive) return;
    const dropped = readArcAssetDragPayload(event.dataTransfer);
    if (!dropped) return;
    const position = pointerCoordinates(event.clientX, event.clientY);
    if (dropped.type === 'material') {
      event.preventDefault();
      event.stopPropagation();
      void assignDroppedMaterialToViewport(window.arc.host, {
        viewportId,
        ...position,
        path: dropped.pathHint,
      }).then((result) => {
        if (result.succeeded) setViewportError('');
        else if (result.error !== 'No scene object at the drop position') setViewportError(result.error);
      });
      return;
    }
    if (dropped.type !== 'mesh' && dropped.type !== 'scene') return;
    event.preventDefault();
    event.stopPropagation();
    void instantiateDroppedMeshInViewport(window.arc.host, {
      viewportId,
      ...position,
      path: dropped.pathHint,
    }).then((result) => {
      if (result.succeeded) setViewportError('');
      else setViewportError(result.error);
    });
  };

  const sendPointer = (event: PointerEvent<HTMLDivElement>, phase: 'down' | 'move' | 'up' | 'leave' | 'cancel') => {
    if (!streamedAvailable) return;
    const position = pointerCoordinates(event.clientX, event.clientY);
    void window.arc.viewport.pointer({
      viewportId,
      phase,
      ...position,
      button: event.button,
      alt: event.altKey,
      shift: event.shiftKey,
      control: event.ctrlKey,
    });
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!viewportAvailable) {
      return;
    }
    event.currentTarget.focus();
    onFocusChange?.(true);
    if (!viewportActive) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    if (event.button === 2) {
      flyNavigationActiveRef.current = true;
      movementLastTickRef.current = performance.now();
    }
    if (streamedAvailable) {
      sendPointer(event, 'down');
      return;
    }
    dragRef.current = {
      pointerId: event.pointerId,
      mode: event.altKey && event.button === 0 ? 'orbit' : event.shiftKey || event.button === 1 ? 'pan' : 'look',
      x: event.clientX,
      y: event.clientY,
    };
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (streamedAvailable) {
      sendPointer(event, 'move');
      return;
    }
    const drag = dragRef.current;
    if (!viewportActive || !drag || drag.pointerId !== event.pointerId) {
      return;
    }
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;
    if (dx === 0 && dy === 0) {
      return;
    }

    if (drag.mode === 'pan') {
      sendCameraInput({ panX: dx, panY: dy });
    } else if (drag.mode === 'orbit') {
      sendCameraInput({ orbitX: dx, orbitY: dy });
    } else {
      sendCameraInput({ lookX: dx, lookY: dy });
    }
  };

  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    if (streamedAvailable) sendPointer(event, 'up');
    if (dragRef.current?.pointerId === event.pointerId) {
      dragRef.current = null;
    }
    if (event.button === 2) {
      flyNavigationActiveRef.current = false;
      movementKeysRef.current.clear();
      movementLastTickRef.current = null;
    }
  };

  const onViewportKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (flyNavigationActiveRef.current && viewportFlyMovementCodes.has(event.code)) {
      event.preventDefault();
      movementKeysRef.current.add(event.code);
      consumedMovementKeysRef.current.add(event.code);
      return;
    }
    if (!streamedAvailable) return;
    void window.arc.viewport.key({
      viewportId,
      key: event.key,
      down: true,
      repeat: event.repeat,
      alt: event.altKey,
      shift: event.shiftKey,
      control: event.ctrlKey,
    });
  };

  const onViewportKeyUp = (event: KeyboardEvent<HTMLDivElement>) => {
    movementKeysRef.current.delete(event.code);
    if (consumedMovementKeysRef.current.delete(event.code)) {
      event.preventDefault();
      return;
    }
    if (!streamedAvailable) return;
    void window.arc.viewport.key({ viewportId, key: event.key, down: false });
  };

  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (!viewportActive) return;
    event.preventDefault();
    const zoom = normalizeViewportWheel(event.deltaY, event.deltaMode);
    if (zoom === 0) return;

    // Wheel zoom does not need pointer coordinates. Route every transport
    // through the same signed camera-input command instead of the streamed
    // pointer queue so zoom-in and zoom-out have identical semantics.
    sendCameraInput({ zoom });
  };

  const frameSelected = () => {
    onCommand('viewport.frameSelected');
    if (viewportActive) {
      sendCameraInput({ focusSelected: true });
    }
  };

  const setGridVisibility = async (visible: boolean) => {
    const previous = gridVisible;
    setLocalGridVisible(visible);
    onGridVisibilityChange?.(visible);
    try {
      const renderOptions = viewportStats.renderOptions ?? defaultRenderOptions;
      const response = (await window.arc.host.command('viewport.setRenderOptions', {
        viewportId,
        ...renderOptions,
        grid: visible,
      })) as ViewportCommandResponse;
      if (!response?.succeeded) throw new Error(response?.error || 'Could not update viewport grid');
      setViewportStats((current) => ({
        ...current,
        renderOptions: { ...(current.renderOptions ?? defaultRenderOptions), grid: visible },
      }));
    } catch (error) {
      setLocalGridVisible(previous);
      onGridVisibilityChange?.(previous);
      setViewportError(error instanceof Error ? error.message : String(error));
    }
  };

  const updateRenderOptions = async (changes: Partial<ViewportRenderOptions>) => {
    const previous = viewportStats.renderOptions ?? defaultRenderOptions;
    const next = { ...previous, ...changes, environment: changes.environment ?? previous.environment };
    setViewportStats((current) => ({ ...current, renderOptions: next }));
    try {
      const response = (await window.arc.host.command('viewport.setRenderOptions', {
        viewportId,
        ...next,
      })) as ViewportCommandResponse;
      if (response?.succeeded === false) throw new Error(response.error || 'Viewport options were rejected');
    } catch (error) {
      setViewportStats((current) => ({ ...current, renderOptions: previous }));
      setViewportError(error instanceof Error ? error.message : String(error));
    }
  };

  const setProjectionMode = async (mode: string) => {
    const camera = viewportStats.camera;
    const focus = camera?.focus.position ?? [0, 0, 0];
    const position = camera?.transform.position ?? [0, 5, 10];
    const distance = Math.max(1, Math.hypot(position[0] - focus[0], position[1] - focus[1], position[2] - focus[2]));
    const direction: Record<string, [number, number, number]> = {
      top: [0, distance, 0],
      bottom: [0, -distance, 0],
      front: [0, 0, distance],
      back: [0, 0, -distance],
      left: [-distance, 0, 0],
      right: [distance, 0, 0],
    };
    try {
      if (mode !== 'perspective') {
        const offset = direction[mode];
        await window.arc.host.command('viewport.setPose', {
          viewportId,
          position: [focus[0] + offset[0], focus[1] + offset[1], focus[2] + offset[2]],
          target: focus,
        });
      }
      const response = (await window.arc.host.command('viewport.setCameraMode', {
        viewportId,
        projection: mode === 'perspective' ? 'perspective' : 'orthographic',
      })) as ViewportCommandResponse;
      if (response?.succeeded === false) throw new Error(response.error || 'Projection was rejected');
      setProjection(mode);
    } catch (error) {
      setViewportError(error instanceof Error ? error.message : String(error));
    }
  };

  const saveBookmark = (slot: number) => {
    if (!viewportStats.camera) return;
    window.localStorage.setItem(`arc.viewport.bookmark.${viewportId}.${slot}`, JSON.stringify(viewportStats.camera));
  };

  const restoreBookmark = async (slot: number) => {
    const stored = window.localStorage.getItem(`arc.viewport.bookmark.${viewportId}.${slot}`);
    if (!stored) return;
    const camera = JSON.parse(stored) as NonNullable<ViewportStats['camera']>;
    await window.arc.host.command('viewport.setPose', {
      viewportId,
      position: camera.transform.position,
      target: camera.focus.position,
    });
  };

  const renderOptions: ViewportRenderOptions = {
    ...defaultRenderOptions,
    ...viewportStats.renderOptions,
    environment: { ...defaultRenderOptions.environment, ...viewportStats.renderOptions?.environment },
  };
  const viewModeLabel =
    renderOptions.renderMode === 'wireframe'
      ? 'Wireframe'
      : renderOptions.visualization === 'lighting'
        ? 'Unlit'
        : 'Lit';
  const transportLabel =
    transport === 'streamed'
      ? 'GPU Shared'
      : transport === 'native'
        ? sharedFailure
          ? 'Native Fallback'
          : 'Native'
        : 'Unavailable';
  const transportTitle =
    transport === 'streamed'
      ? sharedFailure
        ? `Shared GPU viewport failed: ${sharedFailure}`
        : 'Shared GPU texture viewport'
      : transport === 'native' && sharedFailure
        ? `Native child-window fallback. Shared GPU failure: ${sharedFailure}`
        : transport === 'native'
          ? 'Native child-window viewport'
          : 'Viewport unavailable';

  return (
    <section className="arc-viewport-shell">
      <header className="arc-viewport-header">
        <div className="arc-viewport-title">
          <Camera size={14} />
          <span>{viewportId === 'viewport-1' ? 'Viewport 1' : viewportId.replace('viewport-', 'Viewport ')}</span>
        </div>
        <div className="arc-viewport-view-options">
          <details className="arc-viewport-show-menu">
            <summary>
              {projection === 'perspective' ? 'Perspective' : projection[0].toUpperCase() + projection.slice(1)}
            </summary>
            <div className="arc-viewport-show-popup viewport-projection-menu">
              {['perspective', 'top', 'bottom', 'front', 'back', 'left', 'right'].map((mode) => (
                <button key={mode} onClick={() => void setProjectionMode(mode)}>
                  {mode[0].toUpperCase() + mode.slice(1)}
                </button>
              ))}
            </div>
          </details>
          <details className="arc-viewport-show-menu">
            <summary>{viewModeLabel}</summary>
            <div className="arc-viewport-show-popup viewport-mode-menu">
              <button onClick={() => void updateRenderOptions({ renderMode: 'shaded', visualization: 'standard' })}>
                Lit
              </button>
              <button onClick={() => void updateRenderOptions({ renderMode: 'shaded', visualization: 'lighting' })}>
                Unlit
              </button>
              <button onClick={() => void updateRenderOptions({ renderMode: 'wireframe', visualization: 'standard' })}>
                Wireframe
              </button>
              <hr />
              {[
                ['worldNormal', 'Normals'],
                ['albedo', 'Base Color'],
                ['gloss', 'Roughness'],
                ['metalness', 'Metallic'],
                ['lightingHitDistance', 'Depth'],
                ['virtualOverdraw', 'Overdraw'],
                ['lightComplexity', 'Lighting Complexity'],
                ['shadowMask', 'Shadows'],
                ['indirectDiffuse', 'GI'],
                ['reflections', 'Reflections'],
                ['terrainPatchBoundaries', 'Terrain Patches'],
                ['terrainLodLevel', 'Terrain LOD'],
                ['terrainHierarchyNodes', 'Terrain Hierarchy'],
                ['terrainGeometricError', 'Terrain Error'],
                ['terrainCulledNodes', 'Terrain Culling'],
                ['terrainTriangleDensity', 'Terrain Density'],
                ['terrainBounds', 'Terrain Bounds'],
                ['hzbMinimumDepth', 'HZB Minimum Depth'],
                ['hzbMaximumDepth', 'HZB Maximum Depth'],
                ['motionVectors', 'Motion Vectors'],
                ['temporalReactiveMask', 'Reactive Mask'],
                ['temporalDisocclusion', 'Disocclusion'],
                ['temporalConfidence', 'History Confidence'],
                ['temporalRejection', 'History Rejection'],
                ['temporalSampleWeight', 'Temporal Weight'],
                ['textureDesiredMip', 'Texture Desired Mip'],
                ['textureResidentMip', 'Texture Resident Mip'],
                ['virtualTexturePageResidency', 'Virtual Texture Residency'],
                ['virtualTextureRecentRequests', 'Virtual Texture Requests'],
              ].map(([mode, label]) => (
                <button
                  key={mode}
                  onClick={() => void updateRenderOptions({ renderMode: 'shaded', visualization: mode })}
                >
                  {label}
                </button>
              ))}
            </div>
          </details>
          <details className="arc-viewport-show-menu">
            <summary>Show</summary>
            <div className="arc-viewport-show-popup">
              <button
                type="button"
                role="menuitemcheckbox"
                aria-checked={gridVisible}
                onClick={() => void setGridVisibility(!gridVisible)}
              >
                <span className="arc-viewport-menu-check">{gridVisible ? '✓' : ''}</span>
                Grid
              </button>
              <button
                role="menuitemcheckbox"
                aria-checked={renderOptions.shadows}
                onClick={() => void updateRenderOptions({ shadows: !renderOptions.shadows })}
              >
                <span className="arc-viewport-menu-check">{renderOptions.shadows ? '✓' : ''}</span>Shadows
              </button>
              {Object.entries(renderOptions.environment).map(([flag, enabled]) => (
                <button
                  key={flag}
                  role="menuitemcheckbox"
                  aria-checked={enabled}
                  onClick={() =>
                    void updateRenderOptions({ environment: { ...renderOptions.environment, [flag]: !enabled } })
                  }
                >
                  <span className="arc-viewport-menu-check">{enabled ? '✓' : ''}</span>
                  {flag[0].toUpperCase() + flag.slice(1)}
                </button>
              ))}
            </div>
          </details>
        </div>
        <div className="arc-viewport-header-spacer" />
        <div className="arc-viewport-view-options compact">
          <span
            className={`arc-viewport-transport-pill ${transport}${sharedFailure && transport === 'native' ? ' fallback' : ''}`}
            title={transportTitle}
          >
            {transportLabel}
          </span>
          <span className={stats.submitted ? 'arc-viewport-live-pill live' : 'arc-viewport-live-pill'}>
            {stats.submitted ? 'Live' : 'Idle'}
          </span>
          <span className="arc-viewport-header-stat">{formatFps(stats.fps)} FPS</span>
          <span className="arc-viewport-header-stat">{formatFrameTime(stats.frameTimeMs)} ms</span>
          <span className="arc-viewport-header-stat">{formatNumber(stats.drawCalls)} draws</span>
          {stats.width > 0 && stats.height > 0 && (
            <span className="arc-viewport-header-stat">
              {stats.width}x{stats.height}
            </span>
          )}
          <details className="arc-viewport-show-menu">
            <summary>Speed {renderOptions.cameraSpeed}</summary>
            <div className="arc-viewport-show-popup viewport-speed-menu">
              {[0.5, 1, 2, 4, 8, 16].map((speed) => (
                <button key={speed} onClick={() => void updateRenderOptions({ cameraSpeed: speed })}>
                  {speed}x
                </button>
              ))}
            </div>
          </details>
          <details className="arc-viewport-show-menu">
            <summary>
              AA{' '}
              {renderOptions.antiAliasing === 'inherit' ? 'Project/Camera' : renderOptions.antiAliasing.toUpperCase()}
            </summary>
            <div className="arc-viewport-show-popup viewport-aa-menu">
              {(
                [
                  ['inherit', 'Project / Camera Default'],
                  ['disabled', 'Off'],
                  ['fxaa', 'FXAA'],
                  ['taa', 'TAA'],
                  ['taau', 'TAAU'],
                ] as const
              ).map(([method, label]) => (
                <button key={method} onClick={() => void updateRenderOptions({ antiAliasing: method })}>
                  <span className="arc-viewport-menu-check">{renderOptions.antiAliasing === method ? '✓' : ''}</span>
                  {label}
                </button>
              ))}
            </div>
          </details>
          <button
            className={renderOptions.realtime ? 'active' : ''}
            title="Toggle realtime rendering"
            onClick={() => void updateRenderOptions({ realtime: !renderOptions.realtime })}
          >
            {renderOptions.realtime ? <Eye size={13} /> : <EyeOff size={13} />}
          </button>
          <details className="arc-viewport-show-menu">
            <summary>Bookmarks</summary>
            <div className="arc-viewport-show-popup viewport-bookmark-menu">
              {[1, 2, 3, 4].map((slot) => (
                <div className="viewport-bookmark-row" key={slot}>
                  <button onClick={() => void restoreBookmark(slot)}>Recall {slot}</button>
                  <button title={`Save bookmark ${slot}`} onClick={() => saveBookmark(slot)}>
                    +
                  </button>
                </div>
              ))}
            </div>
          </details>
          <details className="arc-viewport-show-menu">
            <summary>Layout</summary>
            <div className="arc-viewport-show-popup viewport-layout-menu">
              {([1, 2, 3, 4] as const).map((count) => (
                <button key={count} onClick={() => onViewportLayoutChange?.(count)}>
                  {count} Viewport{count > 1 ? 's' : ''}
                </button>
              ))}
            </div>
          </details>
          <button title="Frame selected (F), then Alt + Left Drag to orbit" onClick={frameSelected}>
            <Focus size={13} />
          </button>
          <button title="Maximize viewport" onClick={onMaximizeToggle}>
            <Maximize2 size={13} />
          </button>
        </div>
      </header>

      <div
        ref={bodyRef}
        className={viewportActive ? 'arc-viewport-body native-active' : 'arc-viewport-body'}
        onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
            flyNavigationActiveRef.current = false;
            movementKeysRef.current.clear();
            consumedMovementKeysRef.current.clear();
            movementLastTickRef.current = null;
            onFocusChange?.(false);
          }
        }}
        onFocus={() => onFocusChange?.(true)}
        onDragOver={onAssetDragOver}
        onDrop={onAssetDrop}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={(event) => streamedAvailable && sendPointer(event, 'leave')}
        onKeyDown={onViewportKeyDown}
        onKeyUp={onViewportKeyUp}
        onWheel={onWheel}
        onContextMenu={(event) => event.preventDefault()}
        tabIndex={0}
      >
        {streamedAvailable && (
          <canvas id={surfaceId} className="arc-viewport-shared-surface" aria-label="ARC 3D viewport" />
        )}

        {!viewportActive && !viewportAvailable && (
          <div className="arc-viewport-unavailable" role="alert">
            <Box size={32} />
            <h3>Native renderer unavailable</h3>
            <p>{viewportError || startupState?.hostError || 'The ARC native host is not connected.'}</p>
            <button onClick={() => void onReconnect()}>
              <RefreshCw size={14} />
              Reconnect host
            </button>
          </div>
        )}

        {!active && viewportAvailable && (
          <div className="arc-viewport-inactive">
            <Camera size={22} />
            <span>Click to activate {viewportId.replace('viewport-', 'Viewport ')}</span>
          </div>
        )}

        {viewportActive && streamedAvailable && sharedFailure && (
          <div className="arc-viewport-shared-failure" role="alert" onPointerDown={(event) => event.stopPropagation()}>
            <Box size={24} />
            <div className="arc-viewport-shared-failure-copy">
              <strong>Shared GPU viewport failed</strong>
              <span>{sharedFailure}</span>
              <small>Native fallback uses a child window and can render above editor menus.</small>
            </div>
            <div className="arc-viewport-shared-failure-actions">
              <button onClick={retrySharedViewport}>Retry Shared GPU</button>
              <button className="fallback" onClick={() => void activateNativeFallback()}>
                Use Native Fallback
              </button>
            </div>
          </div>
        )}

        {viewportActive && (!streamedAvailable || !sharedFailure) && (viewportError || startupState?.hostError) && (
          <div className="arc-viewport-note">
            <Box size={18} />
            <span>{viewportError || startupState?.hostError}</span>
          </div>
        )}
      </div>
    </section>
  );
}
