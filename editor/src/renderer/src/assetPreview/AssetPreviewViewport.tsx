import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent, ReactNode, WheelEvent } from 'react';

import { normalizeViewportWheel } from '../viewport/viewportWheel';

import './AssetPreviewViewport.css';

type AssetPreviewViewportProps = {
  kind: 'material' | 'shader';
  assetGuid?: string;
  fallback: ReactNode;
  label: string;
};

type ViewportCommandResponse = {
  succeeded?: boolean;
  error?: string;
};

type ViewportStateResponse = {
  succeeded?: boolean;
  error?: string;
  payload?: {
    viewportId?: string;
    submitted?: boolean;
    frameIndex?: number;
    assetPreviewKind?: string;
    assetPreviewGuid?: string;
    assetPreviewError?: string;
  };
};

type ViewportBounds = {
  viewportId: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

type DragState = {
  pointerId: number;
  x: number;
  y: number;
};

const boundsKey = (bounds: ViewportBounds) =>
  `${bounds.viewportId}:${bounds.x}:${bounds.y}:${bounds.width}:${bounds.height}`;

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

export function AssetPreviewViewport({ kind, assetGuid, fallback, label }: AssetPreviewViewportProps) {
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
  const [streamed, setStreamed] = useState(false);
  const [error, setError] = useState('');

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
    return {
      viewportId,
      x: Math.round(rect.left),
      y: Math.round(rect.top),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
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
    let observer: ResizeObserver | null = null;

    const attach = async () => {
      const bounds = currentBounds();
      if (!bounds || bounds.width < 2 || bounds.height < 2 || cancelled) {
        animationFrame = window.requestAnimationFrame(() => void attach());
        return;
      }
      try {
        const response = (await serializeAssetPreviewViewportLifecycle(viewportId, () =>
          window.arc.viewport.create(bounds),
        )) as ViewportCommandResponse | undefined;
        if (response?.succeeded === false) throw new Error(response.error || 'Asset preview surface was rejected');
        if (cancelled) return;
        attachedRef.current = true;
        lastBoundsRef.current = boundsKey(bounds);
        setError('');
        console.info('[material-flow] asset preview viewport attached', { kind, viewportId, guid: normalizedGuid });
        void traceViewportState('attached');
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
      observer?.disconnect();
      pendingBoundsRef.current = null;
      lastBoundsRef.current = '';
      lastPreviewErrorRef.current = '';
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

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!attachedRef.current) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!attachedRef.current || !drag || drag.pointerId !== event.pointerId) return;
    const orbitX = event.clientX - drag.x;
    const orbitY = event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;
    if (orbitX === 0 && orbitY === 0) return;
    void window.arc.viewport
      .cameraInput({ viewportId, orbitX, orbitY })
      .catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)));
  };

  const finishPointer = (event: PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (!attachedRef.current) return;
    event.preventDefault();
    const zoom = normalizeViewportWheel(event.deltaY, event.deltaMode);
    if (!zoom) return;
    void window.arc.viewport
      .cameraInput({ viewportId, zoom })
      .catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)));
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
      <span className="asset-preview-viewport-hint">Drag to orbit · Scroll to zoom</span>
    </div>
  );
}
