import type { CSSProperties, PointerEventHandler, ReactNode } from 'react';

import type { GraphPinDirection, GraphPoint, GraphViewport } from './graphTypes';

export type GraphWire = {
  id: string;
  path: string;
};

export function GraphViewportLayer({
  children,
  className,
  size = 4096,
  viewport,
}: {
  children: ReactNode;
  className?: string;
  size?: number;
  viewport: GraphViewport;
}) {
  return (
    <div
      className={className}
      data-graph-viewport
      style={{
        height: size,
        transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`,
        transformOrigin: '0 0',
        width: size,
      }}
    >
      {children}
    </div>
  );
}

export function GraphWireLayer({
  className,
  pendingPath,
  size = 4096,
  wires,
}: {
  className?: string;
  pendingPath?: string | null;
  size?: number;
  wires: readonly GraphWire[];
}) {
  return (
    <svg aria-hidden="true" className={className} data-graph-wires height={size} width={size}>
      {wires.map((wire) => (
        <path d={wire.path} key={wire.id} />
      ))}
      {pendingPath && <path className="pending" d={pendingPath} />}
    </svg>
  );
}

export function GraphPin({
  className,
  connected,
  direction,
  disabled,
  label,
  onPointerDown,
  pinKey,
  title,
}: {
  className?: string;
  connected?: boolean;
  direction: GraphPinDirection;
  disabled?: boolean;
  label: string;
  onPointerDown?: PointerEventHandler<HTMLButtonElement>;
  pinKey: string;
  title?: string;
}) {
  const classes = [className, direction, connected ? 'connected' : null].filter(Boolean).join(' ');
  const socket = <i data-graph-pin-socket />;
  return (
    <button
      className={classes || undefined}
      data-graph-pin-key={pinKey}
      disabled={disabled}
      onPointerDown={onPointerDown}
      title={title}
      type="button"
    >
      {direction === 'input' ? (
        <>
          {socket} <span>{label}</span>
        </>
      ) : (
        <>
          <span>{label}</span> {socket}
        </>
      )}
    </button>
  );
}

export function GraphSelectionBox({
  className,
  rect,
}: {
  className?: string;
  rect: Pick<CSSProperties, 'left' | 'top' | 'width' | 'height'>;
}) {
  return <div aria-hidden="true" className={className} data-graph-selection-box style={rect} />;
}

export const graphViewportStyle = (viewport: GraphViewport): CSSProperties => ({
  transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`,
});

export const graphPointStyle = (point: GraphPoint, width?: number): CSSProperties => ({
  left: point[0],
  top: point[1],
  ...(width === undefined ? {} : { width }),
});
