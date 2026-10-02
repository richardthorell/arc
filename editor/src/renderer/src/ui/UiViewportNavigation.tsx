import type { HTMLAttributes, ReactNode } from 'react';

import { UiFloatingSurface } from './UiFloatingSurface';

export type UiViewportNavigationProps = Omit<HTMLAttributes<HTMLDivElement>, 'children' | 'role'> & {
  ariaLabel: string;
  children: ReactNode;
};

/** Shared floating surface for viewport navigation controls. */
export function UiViewportNavigation({ ariaLabel, className, children, ...props }: UiViewportNavigationProps) {
  return (
    <UiFloatingSurface
      {...props}
      aria-label={ariaLabel}
      className={['ui-viewport-navigation', className].filter(Boolean).join(' ')}
      role="toolbar"
    >
      {children}
    </UiFloatingSurface>
  );
}
