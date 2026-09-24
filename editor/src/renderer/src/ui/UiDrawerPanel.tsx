import type { ComponentProps } from 'react';

import { UiPanel } from './UiPanel';
import './UiDrawerPanel.css';

export type UiDrawerPanelProps = ComponentProps<typeof UiPanel>;

/**
 * Shared content surface for slide-out utility drawers.
 *
 * The host owns placement, open/closed state, and resize behavior; this component
 * keeps the visual/content contract consistent for Search, AI, Source Control,
 * and future drawer-hosted tools.
 */
export function UiDrawerPanel({ children, className, ...props }: UiDrawerPanelProps) {
  return (
    <UiPanel className={['ui-drawer-panel', className].filter(Boolean).join(' ')} {...props}>
      {children}
    </UiPanel>
  );
}
