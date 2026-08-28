import type { CSSProperties, HTMLAttributes, ReactNode } from 'react';

import './UiFloatingSurface.css';

export type UiFloatingSurfaceProps = Omit<HTMLAttributes<HTMLDivElement>, 'children'> & {
  children: ReactNode;
  width?: CSSProperties['width'];
  maxHeight?: CSSProperties['maxHeight'];
};

export function UiFloatingSurface({
  children,
  className,
  maxHeight,
  onContextMenu,
  onPointerDown,
  onWheel,
  style,
  width,
  ...props
}: UiFloatingSurfaceProps) {
  return (
    <div
      {...props}
      className={['menu-dropdown', 'ui-floating-surface', className].filter(Boolean).join(' ')}
      style={{
        ...(width !== undefined ? { width } : {}),
        ...(maxHeight !== undefined ? { maxHeight } : {}),
        ...style,
      }}
      onContextMenu={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onContextMenu?.(event);
      }}
      onPointerDown={(event) => {
        event.stopPropagation();
        onPointerDown?.(event);
      }}
      onWheel={(event) => {
        event.stopPropagation();
        onWheel?.(event);
      }}
    >
      {children}
    </div>
  );
}
