import type { ButtonHTMLAttributes, CSSProperties, HTMLAttributes, ReactNode } from 'react';

import { UiButton } from './UiButton';

import './UiContextMenu.css';

export type UiContextMenuProps = Omit<HTMLAttributes<HTMLDivElement>, 'children'> & {
  children: ReactNode;
  x?: number;
  y?: number;
  width?: CSSProperties['width'];
  maxHeight?: CSSProperties['maxHeight'];
};

export function UiContextMenu({
  children,
  className,
  maxHeight,
  onContextMenu,
  onPointerDown,
  onWheel,
  style,
  width,
  x,
  y,
  ...props
}: UiContextMenuProps) {
  return (
    <div
      {...props}
      className={['menu-dropdown', 'ui-context-menu', className].filter(Boolean).join(' ')}
      role={props.role ?? 'menu'}
      style={{
        ...(x !== undefined ? { left: x } : {}),
        ...(y !== undefined ? { top: y } : {}),
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
        // Menus own wheel input. Keep native scrolling enabled while preventing
        // parent graph/canvas zoom handlers from seeing the same wheel event.
        event.stopPropagation();
        onWheel?.(event);
      }}
    >
      {children}
    </div>
  );
}

export type UiContextMenuItemProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
  leading?: ReactNode;
  trailing?: ReactNode;
};

export function UiContextMenuItem({ children, className, leading, trailing, ...props }: UiContextMenuItemProps) {
  return (
    <UiButton
      {...props}
      className={['menu-entry', 'ui-context-menu-item', className].filter(Boolean).join(' ')}
      role={props.role ?? 'menuitem'}
      variant="ghost"
    >
      <span className="menu-leading">{leading}</span>
      <span className="menu-entry-label">{children}</span>
      <span className="ui-context-menu-trailing">{trailing}</span>
    </UiButton>
  );
}
