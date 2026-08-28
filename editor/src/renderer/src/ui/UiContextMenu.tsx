import type { ButtonHTMLAttributes, CSSProperties, HTMLAttributes, ReactNode } from 'react';

import { UiButton } from './UiButton';
import { UiFloatingSurface } from './UiFloatingSurface';

import './UiContextMenu.css';

export type UiContextMenuProps = Omit<HTMLAttributes<HTMLDivElement>, 'children'> & {
  children: ReactNode;
  x?: number;
  y?: number;
  width?: CSSProperties['width'];
  maxHeight?: CSSProperties['maxHeight'];
};

export function UiContextMenu({ children, className, maxHeight, style, width, x, y, ...props }: UiContextMenuProps) {
  return (
    <UiFloatingSurface
      {...props}
      className={['ui-context-menu', className].filter(Boolean).join(' ')}
      maxHeight={maxHeight}
      role={props.role ?? 'menu'}
      style={{
        ...(x !== undefined ? { left: x } : {}),
        ...(y !== undefined ? { top: y } : {}),
        ...style,
      }}
      width={width}
    >
      {children}
    </UiFloatingSurface>
  );
}

export type UiContextMenuItemProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
  leading?: ReactNode;
  trailing?: ReactNode;
};

export function UiContextMenuItem({ children, className, leading, trailing, ...props }: UiContextMenuItemProps) {
  const hasLeading = leading !== undefined && leading !== null;

  return (
    <UiButton
      {...props}
      className={[
        'menu-entry',
        'ui-context-menu-item',
        hasLeading ? 'ui-context-menu-item-has-leading' : undefined,
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      role={props.role ?? 'menuitem'}
      variant="ghost"
    >
      {hasLeading && <span className="menu-leading">{leading}</span>}
      <span className="menu-entry-label">{children}</span>
      {trailing !== undefined && trailing !== null && <span className="ui-context-menu-trailing">{trailing}</span>}
    </UiButton>
  );
}
