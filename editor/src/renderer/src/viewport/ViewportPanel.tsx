import { useCallback, useEffect, useRef, useState } from 'react';
import type { PointerEvent, WheelEvent } from 'react';
import { Box, Camera, Eye, Focus, Maximize2, RefreshCw } from 'lucide-react';

import type { CommandId } from '../app/workbenchTypes';
import type { StartupState } from '../app/workbenchTypes';
import type { ProjectSnapshot } from '../services/editorHostTypes';

import './viewport.css';

type ViewportPanelProps = {
  project: ProjectSnapshot | null;
  startupState: StartupState | null;
  onCommand: (command: CommandId) => void;
  onReconnect: () => Promise<void>;
  gridVisible?: boolean;
  onGridVisibilityChange?: (visible: boolean) => void;
  onFocusChange?: (focused: boolean) => void;
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
  width: number;
  height: number;
  fps: number;
  frameTimeMs: number;
  drawCalls: number;
  frameIndex: number;
  submitted: boolean;
  renderOptions?: ViewportRenderOptions;
};

type ViewportRenderOptions = {
  renderMode: 'shaded' | 'wireframe';
  visualization: string;
  overlay: 'none' | 'selectedWireframe' | 'allWireframe';
  shadows: boolean;
  grid: boolean;
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
  environment: { sky: true, fog: true, terrain: true, water: true, vegetation: true, decals: true },
};

type ViewportCommandResponse = {
  succeeded?: boolean;
  error?: string;
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

export function ViewportPanel({
  project,
  startupState,
  onCommand,
  onReconnect,
  gridVisible: controlledGridVisible,
  onGridVisibilityChange,
  onFocusChange,
}: ViewportPanelProps) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const lastAttachAttemptRef = useRef(0);
  const [viewportError, setViewportError] = useState('');
  const [viewportStats, setViewportStats] = useState<ViewportStats>(() => fallbackStats(project));
  const [localGridVisible, setLocalGridVisible] = useState(true);
  const gridVisible = controlledGridVisible ?? localGridVisible;
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
      lastAttachAttemptRef.current = Date.now();
      const response = (await window.arc.viewport.attach(bounds)) as ViewportCommandResponse | undefined;
      if (response?.succeeded === false) throw new Error(response.error || 'Native viewport attachment was rejected');
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
      const response = (await window.arc.viewport.resize(bounds)) as ViewportCommandResponse | undefined;
      if (response?.succeeded === false) throw new Error(response.error || 'Native viewport resize was rejected');
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
        const response = (await window.arc.host.query('viewport.state')) as HostResponse<ViewportStats>;
        if (!cancelled && response?.succeeded && response.payload) {
          setViewportStats(response.payload);
          if (typeof response.payload.renderOptions?.grid === 'boolean')
            setLocalGridVisible(response.payload.renderOptions.grid);
          setViewportError('');
          if (
            (!response.payload.submitted || response.payload.frameIndex === 0) &&
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
  }, [attachViewport, nativeActive, project]);

  const sendCameraInput = (input: Parameters<typeof window.arc.viewport.cameraInput>[0]) => {
    void window.arc.viewport.cameraInput(input).catch((error) => {
      setViewportError(error instanceof Error ? error.message : String(error));
    });
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!nativeActive) {
      return;
    }
    event.currentTarget.focus();
    onFocusChange?.(true);
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      mode: event.altKey && event.button === 0 ? 'orbit' : event.shiftKey || event.button === 1 ? 'pan' : 'look',
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

    if (drag.mode === 'pan') {
      sendCameraInput({ panX: dx, panY: dy });
    } else if (drag.mode === 'orbit') {
      sendCameraInput({ orbitX: dx, orbitY: dy });
    } else {
      sendCameraInput({ lookX: dx, lookY: dy });
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

  const setGridVisibility = async (visible: boolean) => {
    const previous = gridVisible;
    setLocalGridVisible(visible);
    onGridVisibilityChange?.(visible);
    try {
      const renderOptions = viewportStats.renderOptions ?? defaultRenderOptions;
      const response = (await window.arc.host.command('viewport.setRenderOptions', {
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

  return (
    <section className="arc-viewport-shell">
      <header className="arc-viewport-header">
        <div className="arc-viewport-title">
          <Camera size={14} />
          <span>Viewport 1</span>
        </div>
        <div className="arc-viewport-view-options">
          <button title="Frame an object with F, then orbit it with Alt + Left Drag">Perspective</button>
          <button>Lit</button>
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
            </div>
          </details>
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
            <span className="arc-viewport-header-stat">
              {stats.width}x{stats.height}
            </span>
          )}
          <button title="Camera speed">Speed 4</button>
          <button title="Realtime">
            <Eye size={13} />
          </button>
          <button title="Frame selected (F), then Alt + Left Drag to orbit" onClick={frameSelected}>
            <Focus size={13} />
          </button>
          <button title="Maximize">
            <Maximize2 size={13} />
          </button>
        </div>
      </header>

      <div
        ref={bodyRef}
        className={nativeActive ? 'arc-viewport-body native-active' : 'arc-viewport-body'}
        onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) onFocusChange?.(false);
        }}
        onFocus={() => onFocusChange?.(true)}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
        onContextMenu={(event) => event.preventDefault()}
        tabIndex={0}
      >
        {!nativeActive && (
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

        {nativeActive && (viewportError || startupState?.hostError) && (
          <div className="arc-viewport-note">
            <Box size={18} />
            <span>{viewportError || startupState?.hostError}</span>
          </div>
        )}
      </div>
    </section>
  );
}
