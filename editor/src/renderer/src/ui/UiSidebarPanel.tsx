import type { ComponentProps } from 'react';

import { UiPanel } from './UiPanel';
import './UiSidebarPanel.css';

type UiSidebarPanelProps = ComponentProps<typeof UiPanel>;

export function UiSidebarPanel({ children, className, ...props }: UiSidebarPanelProps) {
  return (
    <UiPanel className={['ui-sidebar-panel', className].filter(Boolean).join(' ')} {...props}>
      {children}
    </UiPanel>
  );
}
