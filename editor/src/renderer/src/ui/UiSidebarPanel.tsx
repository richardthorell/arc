import { useEffect, useRef, useState, type ComponentProps, type KeyboardEvent as ReactKeyboardEvent } from 'react';

import { UiPanel } from './UiPanel';
import './UiSidebarPanel.css';

type UiSidebarPanelProps = ComponentProps<typeof UiPanel> & {
  defaultWidth?: number;
  maxWidth?: number;
  minWidth?: number;
  resizeStorageKey?: string;
  resizable?: boolean;
};

const clampWidth = (value: number, minWidth: number, maxWidth: number) =>
  Math.min(maxWidth, Math.max(minWidth, Math.round(value)));

export function UiSidebarPanel({
  children,
  className,
  defaultWidth = 320,
  maxWidth = 640,
  minWidth = 240,
  resizeStorageKey,
  resizable = false,
  style,
  ...props
}: UiSidebarPanelProps) {
  const panel = useRef<HTMLElement | null>(null);
  const initialWidth = () => {
    if (!resizeStorageKey) return clampWidth(defaultWidth, minWidth, maxWidth);
    const saved = Number(window.localStorage.getItem(resizeStorageKey));
    return Number.isFinite(saved) && saved > 0
      ? clampWidth(saved, minWidth, maxWidth)
      : clampWidth(defaultWidth, minWidth, maxWidth);
  };
  const [width, setWidth] = useState(initialWidth);
  const widthRef = useRef(width);
  const [resizing, setResizing] = useState(false);

  const updateWidth = (value: number) => {
    const next = clampWidth(value, minWidth, maxWidth);
    widthRef.current = next;
    setWidth(next);
  };

  const persistWidth = () => {
    if (resizeStorageKey) window.localStorage.setItem(resizeStorageKey, String(widthRef.current));
  };

  useEffect(() => {
    if (!resizing) return;
    const move = (event: PointerEvent) => {
      const bounds = panel.current?.getBoundingClientRect();
      if (bounds) updateWidth(event.clientX - bounds.left);
    };
    const stop = () => {
      setResizing(false);
      persistWidth();
    };

    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stop, { once: true });
    window.addEventListener('pointercancel', stop, { once: true });
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', stop);
      window.removeEventListener('pointercancel', stop);
    };
  }, [resizing]);

  const resizeWithKeyboard = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    let next = widthRef.current;
    if (event.key === 'ArrowLeft') next -= 16;
    else if (event.key === 'ArrowRight') next += 16;
    else if (event.key === 'Home') next = minWidth;
    else if (event.key === 'End') next = maxWidth;
    else return;
    event.preventDefault();
    updateWidth(next);
    persistWidth();
  };

  return (
    <UiPanel
      className={['ui-sidebar-panel', className].filter(Boolean).join(' ')}
      ref={panel}
      style={{ ...style, ...(resizable ? { width } : {}) }}
      {...props}
    >
      {children}
      {resizable && (
        <div
          aria-label="Resize sidebar"
          aria-orientation="vertical"
          aria-valuemax={maxWidth}
          aria-valuemin={minWidth}
          aria-valuenow={width}
          className={`ui-sidebar-panel-resize-handle${resizing ? ' is-resizing' : ''}`}
          onKeyDown={resizeWithKeyboard}
          onPointerDown={(event) => {
            event.preventDefault();
            event.currentTarget.setPointerCapture?.(event.pointerId);
            setResizing(true);
          }}
          role="separator"
          tabIndex={0}
        />
      )}
    </UiPanel>
  );
}
