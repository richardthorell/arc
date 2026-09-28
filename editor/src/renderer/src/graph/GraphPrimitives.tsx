import { useId, useState, type CSSProperties, type PointerEventHandler, type ReactNode } from 'react';

import type { GraphPinDirection, GraphPoint, GraphViewport } from './graphTypes';

export type GraphWire = {
  id: string;
  path: string;
  sourcePinKey?: string;
  destinationPinKey?: string;
  tooltip?: string;
};

export type GraphPinCompatibility = 'compatible' | 'incompatible' | null;

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
      data-graph-zoom={viewport.zoom}
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
  onHoveredWireChange,
  pendingPath,
  size = 4096,
  wires,
}: {
  className?: string;
  onHoveredWireChange?: (wire: GraphWire | null) => void;
  pendingPath?: string | null;
  size?: number;
  wires: readonly GraphWire[];
}) {
  const [hoveredWireId, setHoveredWireId] = useState<string | null>(null);
  const directionPathPrefix = useId().replace(/:/g, '');

  const setHoveredWire = (wire: GraphWire | null) => {
    setHoveredWireId(wire?.id ?? null);
    onHoveredWireChange?.(wire);
  };

  return (
    <svg aria-hidden="true" className={className} data-graph-wires height={size} width={size}>
      {wires.map((wire, index) => {
        const hovered = hoveredWireId === wire.id;
        const directionPathId = `${directionPathPrefix}-wire-${index}`;
        return (
          <g key={wire.id}>
            <path
              className={hovered ? 'is-hovered' : undefined}
              d={wire.path}
              data-destination-pin-key={wire.destinationPinKey}
              data-graph-wire-id={wire.id}
              data-hovered={hovered || undefined}
              data-source-pin-key={wire.sourcePinKey}
              id={directionPathId}
              onPointerEnter={() => setHoveredWire(wire)}
              onPointerLeave={() => setHoveredWire(null)}
            >
              {wire.tooltip ? <title>{wire.tooltip}</title> : null}
            </path>
            {hovered ? (
              <text aria-hidden="true" className="graph-wire-direction" data-graph-wire-direction={wire.id}>
                <textPath href={`#${directionPathId}`} startOffset="38%">
                  › › ›
                </textPath>
              </text>
            ) : null}
          </g>
        );
      })}
      {pendingPath && <path className="pending" d={pendingPath} />}
    </svg>
  );
}

export function GraphPin({
  className,
  compatibility = null,
  connected,
  direction,
  disabled,
  highlighted,
  invalidDrop,
  label,
  onPointerDown,
  pinKey,
  semanticDescription,
  title,
  typeLabel,
}: {
  className?: string;
  compatibility?: GraphPinCompatibility;
  connected?: boolean;
  direction: GraphPinDirection;
  disabled?: boolean;
  highlighted?: boolean;
  invalidDrop?: boolean;
  label: string;
  onPointerDown?: PointerEventHandler<HTMLButtonElement>;
  pinKey: string;
  semanticDescription?: string;
  title?: string;
  typeLabel?: string;
}) {
  const classes = [
    className,
    direction,
    connected ? 'connected' : null,
    highlighted ? 'is-highlighted' : null,
    compatibility === 'compatible' ? 'is-compatible-target' : null,
    compatibility === 'incompatible' ? 'is-incompatible-target' : null,
    invalidDrop ? 'is-invalid-drop' : null,
  ]
    .filter(Boolean)
    .join(' ');
  const socket = <i data-graph-pin-socket />;
  const directionLabel = direction === 'input' ? 'Input' : 'Output';
  const tooltip =
    title ??
    [
      [directionLabel, typeLabel].filter(Boolean).join(' · '),
      semanticDescription ? `${label} — ${semanticDescription}` : label,
    ]
      .filter(Boolean)
      .join(' • ');

  return (
    <button
      className={classes || undefined}
      data-graph-pin-compatibility={compatibility ?? undefined}
      data-graph-pin-direction={direction}
      data-graph-pin-highlighted={highlighted || undefined}
      data-graph-pin-invalid-drop={invalidDrop || undefined}
      data-graph-pin-key={pinKey}
      disabled={disabled}
      onPointerDown={onPointerDown}
      title={tooltip}
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
