import {
  useEffect,
  useRef,
  type ButtonHTMLAttributes,
  type CSSProperties,
  type HTMLAttributes,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';

import { UiButton } from './UiButton';
import { UiFloatingSurface } from './UiFloatingSurface';
import { UiSearchInput } from './UiTextInput';

import './UiContextMenu.css';

export type UiContextMenuProps = Omit<HTMLAttributes<HTMLDivElement>, 'children'> & {
  children: ReactNode;
  x?: number;
  y?: number;
  width?: CSSProperties['width'];
  maxHeight?: CSSProperties['maxHeight'];
  searchValue?: string;
  onSearchValueChange?: (value: string) => void;
  searchPlaceholder?: string;
  searchAriaLabel?: string;
  onRequestClose?: () => void;
  portal?: boolean;
};

export function UiContextMenu({
  children,
  className,
  maxHeight,
  onKeyDown,
  onRequestClose,
  onSearchValueChange,
  portal = false,
  searchAriaLabel = 'Search menu',
  searchPlaceholder = 'Search...',
  searchValue,
  style,
  width,
  x,
  y,
  ...props
}: UiContextMenuProps) {
  const searchRef = useRef<HTMLInputElement | null>(null);
  const searchable = searchValue !== undefined && onSearchValueChange !== undefined;

  useEffect(() => {
    if (searchable) searchRef.current?.focus();
  }, [searchable]);

  const menu = (
    <UiFloatingSurface
      {...props}
      className={[
        'ui-context-menu',
        portal ? 'ui-context-menu-portal' : undefined,
        searchable ? 'ui-context-menu-searchable' : undefined,
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      maxHeight={maxHeight}
      role={props.role ?? 'menu'}
      style={{
        ...(x !== undefined ? { left: x } : {}),
        ...(y !== undefined ? { top: y } : {}),
        ...style,
      }}
      width={width}
      onKeyDown={(event) => {
        if (event.key === 'Escape') {
          event.preventDefault();
          event.stopPropagation();
          if (searchable && searchValue.length > 0) {
            onSearchValueChange('');
            return;
          }
          onRequestClose?.();
          return;
        }
        onKeyDown?.(event);
      }}
    >
      {searchable && (
        <div className="ui-context-menu-search">
          <UiSearchInput
            ref={searchRef}
            aria-label={searchAriaLabel}
            placeholder={searchPlaceholder}
            value={searchValue}
            onChange={(event) => onSearchValueChange(event.target.value)}
          />
        </div>
      )}
      {children}
    </UiFloatingSurface>
  );

  return portal ? createPortal(menu, document.body) : menu;
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
