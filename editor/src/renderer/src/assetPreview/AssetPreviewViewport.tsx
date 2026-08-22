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

export function assetPreviewViewportId(kind: AssetPreviewViewportProps['kind'], assetGuid: string) {
  return `asset-preview-${kind}-${normalizedAssetGuid(assetGuid)}`;
}

export function AssetPreviewViewport({ kind, assetGuid, fallback, label }: AssetPreviewViewportProps) {
  const normalizedGuid = normalizedAssetGuid(assetGuid);
  const viewportId = useMemo(
    () => (normalizedGuid ? assetPreviewViewportId(kind, normalizedGuid) : ''),
    [kind, normalizedGuid],
  );
  const surfaceId = useMemo(
    () => `arc-asset-preview-surface-${viewportId.replaceAll(/[^a-zA-Z0-9_-]/g, '-')}`,
    [viewportId],
  );
  const rootRef = useRef<HTMLDivElement | null>(null);
  const attachedRef = useRef(false);
  const lastBoundsRef = useRef('');
  const resizeInFlightRef = useRef(false);
  const pendingBoundsRef = useRef<ViewportBounds | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [streamed, setStreamed] = useState(false);
  const [error, setError] = useState('');

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
    let observer: ResizeObserver | null = null;

    const attach = async () => {
      const bounds = currentBounds();
      if (!bounds || bounds.width < 2 || bounds.height < 2 || cancelled) {
        animationFrame = window.requestAnimationFrame(() => void attach());
        return;
      }
      try {
        const response = (await window.arc.viewport.create(bounds)) as ViewportCommandResponse | undefined;
        if (response?.succeeded === false) throw new Error(response.error || 'Asset preview surface was rejected');
        if (cancelled) {
          void window.arc.viewport.detach?.(viewportId);
          return;
        }
        attachedRef.current = true;
        lastBoundsRef.current = boundsKey(bounds);
        setError('');
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
      observer?.disconnect();
      pendingBoundsRef.current = null;
      lastBoundsRef.current = '';
      if (attachedRef.current) void window.arc.viewport.detach?.(viewportId);
      attachedRef.current = false;
    };
  }, [currentBounds, resize, streamed, viewportId]);

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
    void window.arc.viewport.cameraInput({ viewportId, orbitX, orbitY }).catch((reason) =>
      setError(reason instanceof Error ? reason.message : String(reason)),
    );
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
    void window.arc.viewport.cameraInput({ viewportId, zoom }).catch((reason) =>
      setError(reason instanceof Error ? reason.message : String(reason)),
    );
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
