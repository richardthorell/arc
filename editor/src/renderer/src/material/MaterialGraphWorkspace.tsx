import type { CSSProperties, KeyboardEvent, PointerEvent, ReactNode } from 'react';

export type MaterialGraphWorkspaceProps = {
  graph: ReactNode;
  sidebar?: ReactNode;
  sidebarWidth?: number;
  minimumGraphWidth?: number;
  dividerWidth?: number;
  dividerLabel?: string;
  dividerValueMin?: number;
  dividerValueMax?: number;
  onDividerPointerDown?: (event: PointerEvent<HTMLDivElement>) => void;
  onDividerPointerMove?: (event: PointerEvent<HTMLDivElement>) => void;
  onDividerPointerUp?: (event: PointerEvent<HTMLDivElement>) => void;
  onDividerPointerCancel?: (event: PointerEvent<HTMLDivElement>) => void;
  onDividerDoubleClick?: () => void;
  onDividerKeyDown?: (event: KeyboardEvent<HTMLDivElement>) => void;
  className?: string;
};

/**
 * Shared document workspace for Material and Material Function graph assets.
 *
 * The graph surface is common. Asset-specific UI belongs in the optional sidebar:
 * Materials provide preview/render-output settings; Material Functions provide
 * only function metadata/signature authoring.
 */
export function MaterialGraphWorkspace({
  graph,
  sidebar,
  sidebarWidth = 420,
  minimumGraphWidth = 520,
  dividerWidth = 5,
  dividerLabel = 'Resize graph sidebar',
  dividerValueMin = 320,
  dividerValueMax = 760,
  onDividerPointerDown,
  onDividerPointerMove,
  onDividerPointerUp,
  onDividerPointerCancel,
  onDividerDoubleClick,
  onDividerKeyDown,
  className,
}: MaterialGraphWorkspaceProps) {
  const hasSidebar = sidebar !== undefined && sidebar !== null;
  const style: CSSProperties = hasSidebar
    ? { gridTemplateColumns: `minmax(${minimumGraphWidth}px, 1fr) ${dividerWidth}px ${sidebarWidth}px` }
    : { gridTemplateColumns: 'minmax(0, 1fr)' };

  return (
    <section className={['material-editor', 'material-graph-workspace', className].filter(Boolean).join(' ')} style={style}>
      <div className="material-editor-graph-region">{graph}</div>

      {hasSidebar && (
        <>
          <div
            aria-label={dividerLabel}
            aria-orientation="vertical"
            aria-valuemax={dividerValueMax}
            aria-valuemin={dividerValueMin}
            aria-valuenow={sidebarWidth}
            className="material-editor-divider"
            role={onDividerPointerDown || onDividerKeyDown ? 'separator' : undefined}
            tabIndex={onDividerPointerDown || onDividerKeyDown ? 0 : undefined}
            style={{
              cursor: onDividerPointerDown ? 'col-resize' : undefined,
              touchAction: onDividerPointerDown ? 'none' : undefined,
              borderLeft: '1px solid rgba(102, 132, 146, 0.14)',
              borderRight: '1px solid rgba(102, 132, 146, 0.22)',
              background: '#0e171c',
            }}
            onDoubleClick={onDividerDoubleClick}
            onKeyDown={onDividerKeyDown}
            onPointerCancel={onDividerPointerCancel}
            onPointerDown={onDividerPointerDown}
            onPointerMove={onDividerPointerMove}
            onPointerUp={onDividerPointerUp}
          />
          <aside className="material-editor-sidebar editor-property-panel">{sidebar}</aside>
        </>
      )}
    </section>
  );
}
