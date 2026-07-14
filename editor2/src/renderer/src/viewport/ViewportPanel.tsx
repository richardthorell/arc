import { useCallback, useEffect, useRef, useState } from 'react';
import type { PointerEvent, WheelEvent } from 'react';
import {
  Box,
  Camera,
  Crosshair,
  Eye,
  Focus,
  Grid3X3,
  Hand,
  Maximize2,
  MousePointer2,
  Move3D,
  Orbit,
  RotateCw,
  Scaling,
} from 'lucide-react';

import type { CommandId } from '../app/workbenchTypes';
import type { StartupState } from '../app/workbenchTypes';
import type { ProjectSnapshot } from '../services/mockHost';

import './viewport.css';

type ViewportPanelProps = {
  project: ProjectSnapshot | null;
  startupState: StartupState | null;
  onCommand: (command: CommandId) => void;
};

type DragState = {
  pointerId: number;
  button: number;
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
  width: number;
  height: number;
  fps: number;
  frameTimeMs: number;
  drawCalls: number;
  frameIndex: number;
  submitted: boolean;
};

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

export function ViewportPanel({ project, startupState, onCommand }: ViewportPanelProps) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [viewportError, setViewportError] = useState('');
  const [viewportStats, setViewportStats] = useState<ViewportStats>(() => fallbackStats(project));
  const nativeActive = startupState?.viewportMode === 'native' && Boolean(window.arc?.viewport);
  const stats = nativeActive ? viewportStats : fallbackStats(project);

  const viewportBounds = useCallback(() => {
    const element = bodyRef.current;
    if (!element) {
      return null;
    }
    const rect = element.getBoundingClientRect();
    return {
      x: rect.left,
      y: rect.top,
      width: rect.width,
      height: rect.height,
    };
  }, []);

  const attachViewport = useCallback(async () => {
    if (!nativeActive) {
      return;
    }
    const bounds = viewportBounds();
    if (!bounds || bounds.width < 2 || bounds.height < 2) {
      return;
    }
    try {
      await window.arc.viewport.attach(bounds);
      setViewportError('');
    } catch (error) {
      setViewportError(error instanceof Error ? error.message : String(error));
    }
  }, [nativeActive, viewportBounds]);

  const resizeViewport = useCallback(async () => {
    if (!nativeActive) {
      return;
    }
    const bounds = viewportBounds();
    if (!bounds || bounds.width < 2 || bounds.height < 2) {
      return;
    }
    try {
      await window.arc.viewport.resize(bounds);
      setViewportError('');
    } catch (error) {
      setViewportError(error instanceof Error ? error.message : String(error));
    }
  }, [nativeActive, viewportBounds]);

  useEffect(() => {
    void attachViewport();
  }, [attachViewport]);

  useEffect(() => {
    const element = bodyRef.current;
    if (!element || !nativeActive) {
      return;
    }

    const observer = new ResizeObserver(() => {
      void resizeViewport();
    });
    observer.observe(element);
    const interval = window.setInterval(resizeViewport, 1000);
    window.addEventListener('resize', resizeViewport);
    return () => {
      window.clearInterval(interval);
      observer.disconnect();
      window.removeEventListener('resize', resizeViewport);
    };
  }, [nativeActive, resizeViewport]);

  useEffect(() => {
    if (!nativeActive || !window.arc?.host) {
      setViewportStats(fallbackStats(project));
      return;
    }

    let cancelled = false;
    const pollStats = async () => {
      try {
        const response = await window.arc.host.query('viewport.state') as HostResponse<ViewportStats>;
        if (!cancelled && response?.succeeded && response.payload) {
          setViewportStats(response.payload);
          setViewportError('');
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
  }, [nativeActive, project]);

  const sendCameraInput = (input: Parameters<typeof window.arc.viewport.cameraInput>[0]) => {
    void window.arc.viewport.cameraInput(input).catch((error) => {
      setViewportError(error instanceof Error ? error.message : String(error));
    });
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!nativeActive) {
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      button: event.button,
      x: event.clientX,
      y: event.clientY,
    };
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!nativeActive || !drag || drag.pointerId !== event.pointerId) {
      return;
    }
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;
    if (dx === 0 && dy === 0) {
      return;
    }

    if (event.altKey) {
      sendCameraInput({ forward: dy });
    } else if (event.shiftKey || drag.button === 1) {
      sendCameraInput({ panX: dx, panY: dy });
    } else {
      sendCameraInput({ orbitX: dx, orbitY: dy });
    }
  };

  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) {
      dragRef.current = null;
    }
  };

  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (!nativeActive) {
      return;
    }
    sendCameraInput({ zoom: -event.deltaY / 120 });
  };

  const frameSelected = () => {
    onCommand('viewport.frameSelected');
    if (nativeActive) {
      sendCameraInput({ focusSelected: true });
    }
  };

  return (
    <section className="arc-viewport-shell">
      <header className="arc-viewport-header">
        <div className="arc-viewport-title">
          <Camera size={14} />
          <span>Viewport 1</span>
        </div>
        <div className="arc-viewport-view-options">
          <button>Perspective</button>
          <button>Lit</button>
          <button>Show</button>
        </div>
        <div className="arc-viewport-header-spacer" />
        <div className="arc-viewport-view-options compact">
          <span className={stats.submitted ? 'arc-viewport-live-pill live' : 'arc-viewport-live-pill'}>
            {stats.submitted ? 'Live' : 'Idle'}
          </span>
          <span className="arc-viewport-header-stat">{formatFps(stats.fps)} FPS</span>
          <span className="arc-viewport-header-stat">{formatFrameTime(stats.frameTimeMs)} ms</span>
          <span className="arc-viewport-header-stat">{formatNumber(stats.drawCalls)} draws</span>
          {stats.width > 0 && stats.height > 0 && (
            <span className="arc-viewport-header-stat">{stats.width}x{stats.height}</span>
          )}
          <button title="Camera speed">Speed 4</button>
          <button title="Realtime"><Eye size={13} /></button>
          <button title="Maximize"><Maximize2 size={13} /></button>
        </div>
      </header>

      <div
        ref={bodyRef}
        className={nativeActive ? 'arc-viewport-body native-active' : 'arc-viewport-body'}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
        onContextMenu={(event) => event.preventDefault()}
      >
        {!nativeActive && (
          <>
            <div className="arc-viewport-scene-bg" />
            <div className="arc-viewport-depth-fog" />
            <div className="arc-viewport-terrain" />
            <div className="arc-viewport-object-shadow" />
            <div className="arc-viewport-selected-object">
              <div className="arc-selected-roof" />
              <div className="arc-selected-body" />
              <div className="arc-selection-outline" />
            </div>
            <div className="arc-transform-gizmo" aria-label="Transform gizmo">
              <span className="axis-y" />
              <span className="axis-x" />
              <span className="axis-z" />
              <strong />
            </div>
          </>
        )}

        {!nativeActive && <aside className="arc-viewport-tool-strip">
          <button title="Select" onClick={() => onCommand('viewport.select')}><MousePointer2 size={16} /></button>
          <button title="Pan"><Hand size={16} /></button>
          <button className="active" title="Translate" onClick={() => onCommand('viewport.translate')}><Move3D size={16} /></button>
          <button title="Rotate" onClick={() => onCommand('viewport.rotate')}><RotateCw size={16} /></button>
          <button title="Scale" onClick={() => onCommand('viewport.scale')}><Scaling size={16} /></button>
          <button title="Frame selected" onClick={frameSelected}><Focus size={16} /></button>
        </aside>}

        {!nativeActive && <aside className="arc-viewport-stats">
          <dl>
            <div><dt>FPS</dt><dd>{formatFps(stats.fps)}</dd></div>
            <div><dt>Frame</dt><dd>{formatFrameTime(stats.frameTimeMs)} ms</dd></div>
            <div><dt>Draw Calls</dt><dd>{formatNumber(stats.drawCalls)}</dd></div>
            <div><dt>Mode</dt><dd>Placeholder</dd></div>
          </dl>
        </aside>}

        {!nativeActive && <div className="arc-viewport-overlay-top-left">
          <span><Grid3X3 size={13} /> Grid</span>
          <span><Crosshair size={13} /> Snapping 0.25</span>
          <span><Orbit size={13} /> Global</span>
        </div>}

        {!nativeActive && <div className="arc-viewport-breadcrumb">
          <span>World</span>
          <span>Buildings</span>
          <span>Cabin_01</span>
          <strong>SM_Cabin</strong>
        </div>}

        {!nativeActive && <div className="arc-axis-gizmo-large">
          <span className="axis-label-y">Y</span>
          <span className="axis-label-x">X</span>
          <span className="axis-label-z">Z</span>
        </div>}

        {(!nativeActive || viewportError || startupState?.hostError) && <div className="arc-viewport-note">
          <Box size={18} />
          <span>{viewportError || startupState?.hostError || 'Viewport shell only. Native engine rendering is not connected.'}</span>
        </div>}
      </div>
    </section>
  );
}
