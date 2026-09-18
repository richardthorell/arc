import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent, ReactNode, WheelEvent } from 'react';

import { normalizeViewportWheel } from '../viewport/viewportWheel';
import {
  clampMaterialPreviewOrbitY,
  clampMaterialPreviewZoom,
  materialPreviewDefaultCameraDistance,
  materialPreviewInitialCameraPitch,
  materialPreviewInitialZoom,
  materialPreviewNativeCameraDistance,
} from './materialPreviewCamera';

import './AssetPreviewViewport.css';

type ModelPreviewMesh = {
  name?: string;
  skinned?: boolean;
};

type ModelPreviewSkeleton = {
  name?: string;
  boneCount?: number;
  hierarchyDepth?: number;
  rootBone?: string;
  joints?: Array<{ index: number; name: string; parent: number }>;
};

type ViewportStatePayload = {
  viewportId?: string;
  submitted?: boolean;
  frameIndex?: number;
  assetPreviewKind?: string;
  assetPreviewGuid?: string;
  assetPreviewError?: string;
  assetPreviewReady?: boolean;
  modelMeshes?: ModelPreviewMesh[];
  modelSkeleton?: ModelPreviewSkeleton;
};

type AssetPreviewViewportProps = {
  kind: 'material' | 'shader' | 'model';
  assetGuid?: string;
  fallback: ReactNode;
  label: string;
  materialMesh?: 'sphere' | 'cube' | 'pill';
  materialAutoRotate?: boolean;
  loading?: boolean;
  onState?: (payload: ViewportStatePayload | undefined) => void;
};

type ViewportCommandResponse = {
  succeeded?: boolean;
  error?: string;
};

type ViewportStateResponse = {
  succeeded?: boolean;
  error?: string;
  payload?: ViewportStatePayload;
};

type ViewportBounds = {
  viewportId: string;
  x: number;
  y: number;
  width: number;
  height: number;
  devicePixelRatio: number;
};

type DragState = {
  pointerId: number;
  x: number;
  y: number;
};

const boundsKey = (bounds: ViewportBounds) =>
  `${bounds.viewportId}:${bounds.x}:${bounds.y}:${bounds.width}:${bounds.height}:${bounds.devicePixelRatio}`;

const normalizedAssetGuid = (guid?: string) => guid?.trim().toLowerCase() ?? '';

const assetPreviewViewportLifecycle = new Map<string, Promise<void>>();
let nextAssetPreviewViewportInstance = 1;

export function serializeAssetPreviewViewportLifecycle<T>(viewportId: string, operation: () => Promise<T>): Promise<T> {
  const previous = assetPreviewViewportLifecycle.get(viewportId) ?? Promise.resolve();
  const result = previous.catch(() => undefined).then(operation);
  const tail = result.then(
    () => undefined,
    () => undefined,
  );
  assetPreviewViewportLifecycle.set(viewportId, tail);
  void tail.finally(() => {
    if (assetPreviewViewportLifecycle.get(viewportId) === tail) assetPreviewViewportLifecycle.delete(viewportId);
  });
  return result;
}

export function assetPreviewViewportId(kind: AssetPreviewViewportProps['kind'], assetGuid: string, instance?: number) {
  const base = `asset-preview-${kind}-${normalizedAssetGuid(assetGuid)}`;
  return instance === undefined ? base : `${base}~${instance}`;
}

export function materialPreviewRenderOptions() {
  return {
    renderMode: 'shaded',
    visualization: 'standard',
    overlay: 'none',
    selectionOutline: false,
    hoverOutline: false,
    selectionBounds: false,
    componentGizmos: false,
    selectionHierarchy: false,
    shadows: true,
    grid: false,
    skeletons: false,
    realtime: true,
    environment: {
      sky: true,
      fog: false,
      terrain: false,
      water: false,
      vegetation: false,
      decals: false,
    },
  } as const;
}

export function AssetPreviewViewport({
  kind,
  assetGuid,
  fallback,
  label,
  materialMesh = 'sphere',
  materialAutoRotate = true,
  loading = false,
  onState,
}: AssetPreviewViewportProps) {
  const normalizedGuid = normalizedAssetGuid(assetGuid);
  const viewportInstanceRef = useRef<number | null>(null);
  if (viewportInstanceRef.current === null) viewportInstanceRef.current = nextAssetPreviewViewportInstance++;
  const viewportInstance = viewportInstanceRef.current;
  const viewportId = useMemo(
    () => (normalizedGuid ? assetPreviewViewportId(kind, normalizedGuid, viewportInstance) : ''),
    [kind, normalizedGuid, viewportInstance],
  );
  const surfaceId = useMemo(
    () => `arc-asset-preview-surface-${viewportId.replaceAll(/[^a-zA-Z0-9_-]/g, '-')}`,
    [viewportId],
  );
  const rootRef = useRef<HTMLDivElement | null>(null);
  const attachedRef = useRef(false);
  const lastBoundsRef = useRef('');
  const lastPreviewErrorRef = useRef('');
  const resizeInFlightRef = useRef(false);
  const pendingBoundsRef = useRef<ViewportBounds | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const materialCameraDistanceRef = useRef(materialPreviewNativeCameraDistance);
  const materialCameraPitchRef = useRef(materialPreviewInitialCameraPitch);
  const materialMeshRef = useRef(materialMesh);
  const materialAutoRotateRef = useRef(materialAutoRotate);
  const onStateRef = useRef(onState);
  const [streamed, setStreamed] = useState(false);
  const [previewReady, setPreviewReady] = useState(kind !== 'material');
  const [error, setError] = useState('');

  useEffect(() => {
    onStateRef.current = onState;
  }, [onState]);

  useEffect(() => {
    materialMeshRef.current = materialMesh;
    materialAutoRotateRef.current = materialAutoRotate;
    if (kind !== 'material' || !attachedRef.current || !viewportId) return;

    void window.arc.host
      .command('viewport.setRenderOptions', {
        viewportId,
        ...materialPreviewRenderOptions(),
        materialPreviewMesh: materialMesh,
        materialPreviewAutoRotate,
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, [kind, materialAutoRotate, materialMesh, viewportId]);

  const traceViewportState = useCallback(
    async (phase: string) => {
      if (!viewportId || !window.arc?.host?.query) return;
      try {
        const response = (await window.arc.host.query('viewport.state', { viewportId })) as ViewportStateResponse;
        const previewError = response?.payload?.assetPreviewError ?? '';
        console.info('[material-flow] asset preview viewport state', {
          phase,
          kind,
          viewportId,
          guid: normalizedGuid,
          succeeded: response?.succeeded ?? false,
          error: response?.error ?? '',
          submitted: response?.payload?.submitted ?? false,
          frameIndex: response?.payload?.frameIndex ?? 0,
          assetPreviewError: previewError,
        });
        if (kind === 'material' && previewError && previewError !== lastPreviewErrorRef.current) {
          console.error('[material-flow] material preview realization failed', {
            phase,
            viewportId,
            guid: normalizedGuid,
            error: previewError,
          });
        } else if (kind === 'material' && !previewError && lastPreviewErrorRef.current) {
          console.info('[material-flow] material preview realization recovered', {
            phase,
            viewportId,
            guid: normalizedGuid,
          });
        }
        lastPreviewErrorRef.current = previewError;
        onStateRef.current?.(response?.payload);
      } catch (reason) {
        console.warn('[material-flow] asset preview viewport state query failed', {
          phase,
          kind,
          viewportId,
          reason: reason instanceof Error ? reason.message : String(reason),
        });
      }
    },
    [kind, normalizedGuid, viewportId],
  );

  const currentBounds = useCallback((): ViewportBounds | null => {
    if (!viewportId) return null;
    const element = rootRef.current;
    if (!element) return null;
    const rect = element.getBoundingClientRect();
    const devicePixelRatio =
      Number.isFinite(window.devicePixelRatio) && window.devicePixelRatio > 0
        ? Number(window.devicePixelRatio.toFixed(4))
        : 1;
    return {
      viewportId,
      x: Math.round(rect.left),
      y: Math.round(rect.top),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      devicePixelRatio,
    };
  }, [viewportId]);

  const resize = useCallback(() => {
    if (!attachedRef.current) return;
    const bounds = currentBounds();
    if (!bounds || bounds.width < 2 || bounds.height < 2) return;
    const key = boundsKey(bounds);
    if (key === lastBoundsRef.current) return;
    lastBoundsRef.current = key;
    pendingBoundsRef.current = bounds;
    if (resizeInFlightRef.current) return;

    resizeInFlightRef.current = true;
    void (async () => {
      try {
        while (pendingBoundsRef.current) {
          const next = pendingBoundsRef.current;
          pendingBoundsRef.current = null;
          const response = (await window.arc.viewport.resize(next)) as ViewportCommandResponse | undefined;
          if (response?.succeeded === false) throw new Error(response.error || 'Asset preview resize was rejected');
        }
      } catch (reason) {
        lastBoundsRef.current = '';
        setError(reason instanceof Error ? reason.message : String(reason));
      } finally {
        resizeInFlightRef.current = false;
      }
    })();
  }, [currentBounds]);

  useEffect(() => {
    let cancelled = false;
    void window.arc
      .getStartupState()
      .then((state) => {
        if (!cancelled) setStreamed(state.engineHostConnected && state.viewportMode === 'streamed');
      })
      .catch((reason) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!streamed || !viewportId) return;
    window.arc.viewport.registerSurface?.(viewportId, surfaceId);
    return () => window.arc.viewport.unregisterSurface?.(viewportId);
  }, [streamed, surfaceId, viewportId]);

  useEffect(() => {
    if (!streamed || !viewportId) return;
    let cancelled = false;
    let animationFrame = 0;
    let traceTimer = 0;
    let readyTimer = 0;
    let observer: ResizeObserver | null = null;

    const pollPreviewReady = async () => {
      if (cancelled || kind !== 'material') return;
      try {
        const response = (await window.arc.host.query('viewport.state', { viewportId })) as ViewportStateResponse;
        const payload = response?.payload;
        onStateRef.current?.(payload);
        if (payload?.assetPreviewReady === true) {
          if (!cancelled) setPreviewReady(true);
          return;
        }
      } catch {
        // The viewport may still be attaching. Keep the loading cover up and retry.
      }
      if (!cancelled) readyTimer = window.setTimeout(() => void pollPreviewReady(), 80);
    };

    const attach = async () => {
      const bounds = currentBounds();
      if (!bounds || bounds.width < 2 || bounds.height < 2 || cancelled) {
        animationFrame = window.requestAnimationFrame(() => void attach());
        return;
      }
      try {
        if (kind === 'material') setPreviewReady(false);
        const response = (await serializeAssetPreviewViewportLifecycle(viewportId, async () => {
          const created = (await window.arc.viewport.create(bounds)) as ViewportCommandResponse | undefined;
          if (created?.succeeded === false) return created;
          if (kind !== 'material') return created;
          const configured = (await window.arc.host.command('viewport.setRenderOptions', {
            viewportId,
            ...materialPreviewRenderOptions(),
            materialPreviewMesh: materialMeshRef.current,
            materialPreviewAutoRotate: materialAutoRotateRef.current,
          })) as ViewportCommandResponse | undefined;
          if (configured?.succeeded === false)
            throw new Error(configured.error || 'Material preview render options were rejected');

          materialCameraDistanceRef.current = materialPreviewNativeCameraDistance;
          materialCameraPitchRef.current = materialPreviewInitialCameraPitch;
          const framed = (await window.arc.viewport.cameraInput({
            viewportId,
            zoom: materialPreviewInitialZoom,
          })) as ViewportCommandResponse | undefined;
          if (framed?.succeeded === false)
            throw new Error(framed.error || 'Material preview camera framing was rejected');
          materialCameraDistanceRef.current = materialPreviewDefaultCameraDistance;
          return created;
        })) as ViewportCommandResponse | undefined;
        if (response?.succeeded === false) throw new Error(response.error || 'Asset preview surface was rejected');
        if (cancelled) return;
        attachedRef.current = true;
        lastBoundsRef.current = boundsKey(bounds);
        setError('');
        console.info('[material-flow] asset preview viewport attached', { kind, viewportId, guid: normalizedGuid });
        void traceViewportState('attached');
        if (kind === 'material') void pollPreviewReady();
        traceTimer = window.setTimeout(() => void traceViewportState('after-first-frame'), 150);
        observer = new ResizeObserver(resize);
        if (rootRef.current) observer.observe(rootRef.current);
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      }
    };

    void attach();
    return () => {
      cancelled = true;
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
      if (traceTimer) window.clearTimeout(traceTimer);
      if (readyTimer) window.clearTimeout(readyTimer);
      observer?.disconnect();
      pendingBoundsRef.current = null;
      lastBoundsRef.current = '';
      lastPreviewErrorRef.current = '';
      materialCameraDistanceRef.current = materialPreviewNativeCameraDistance;
      materialCameraPitchRef.current = materialPreviewInitialCameraPitch;
      if (attachedRef.current) {
        console.info('[material-flow] asset preview viewport detaching', { kind, viewportId, guid: normalizedGuid });
        void traceViewportState('before-detach');
      }
      void serializeAssetPreviewViewportLifecycle(viewportId, async () => {
        await window.arc.viewport.detach?.(viewportId);
      });
      attachedRef.current = false;
    };
  }, [currentBounds, kind, normalizedGuid, resize, streamed, traceViewportState, viewportId]);

  const previewIsLoading = kind === 'material' && (loading || !previewReady);

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!attachedRef.current || previewIsLoading) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!attachedRef.current || previewIsLoading || !drag || drag.pointerId !== event.pointerId) return;
    const orbitX = event.clientX - drag.x;
    let orbitY = event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;

    const previousMaterialPitch = materialCameraPitchRef.current;
    let nextMaterialPitch: number | undefined;
    if (kind === 'material') {
      const clamped = clampMaterialPreviewOrbitY(previousMaterialPitch, orbitY);
      orbitY = clamped.orbitY;
      nextMaterialPitch = clamped.pitch;
      materialCameraPitchRef.current = clamped.pitch;
    }

    if (orbitX === 0 && orbitY === 0) return;
    void window.arc.viewport.cameraInput({ viewportId, orbitX, orbitY }).catch((reason) => {
      if (kind === 'material' && nextMaterialPitch === materialCameraPitchRef.current)
        materialCameraPitchRef.current = previousMaterialPitch;
      setError(reason instanceof Error ? reason.message : String(reason));
    });
  };

  const finishPointer = (event: PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (!attachedRef.current || previewIsLoading) return;
    event.preventDefault();
    let zoom = normalizeViewportWheel(event.deltaY, event.deltaMode);
    if (!zoom) return;

    const previousMaterialDistance = materialCameraDistanceRef.current;
    let nextMaterialDistance: number | undefined;
    if (kind === 'material') {
      const clamped = clampMaterialPreviewZoom(previousMaterialDistance, zoom);
      zoom = clamped.zoom;
      nextMaterialDistance = clamped.distance;
      if (!zoom) return;
      materialCameraDistanceRef.current = clamped.distance;
    }

    void window.arc.viewport.cameraInput({ viewportId, zoom }).catch((reason) => {
      if (kind === 'material' && nextMaterialDistance === materialCameraDistanceRef.current)
        materialCameraDistanceRef.current = previousMaterialDistance;
      setError(reason instanceof Error ? reason.message : String(reason));
    });
  };

  if (!normalizedGuid || !streamed || error) {
    return (
      <div className="asset-preview-viewport-fallback" title={error || undefined}>
        {fallback}
      </div>
    );
  }

  return (
    <div
      ref={rootRef}
      className="asset-preview-viewport"
      aria-label={label}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={finishPointer}
      onPointerCancel={finishPointer}
      onWheel={onWheel}
      role="img"
    >
      <canvas id={surfaceId} className="asset-preview-viewport-canvas" aria-hidden="true" />
      {previewIsLoading && (
        <div className="asset-preview-viewport-loading" role="status" aria-live="polite">
          <span className="asset-preview-viewport-loading-spinner" aria-hidden="true" />
          <strong>Loading preview</strong>
          <span>Preparing material and studio lighting…</span>
        </div>
      )}
      <span className="asset-preview-viewport-hint">Drag to orbit · Scroll to zoom</span>
    </div>
  );
}
