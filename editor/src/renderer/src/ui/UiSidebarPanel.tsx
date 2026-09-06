import type { HTMLAttributes, ReactNode } from 'react';

import { UiPanel } from './UiPanel';
import './UiSidebarPanel.css';

type UiSidebarPanelProps = HTMLAttributes<HTMLElement> & {
  actions?: ReactNode;
  children: ReactNode;
  icon?: ReactNode;
  subtitle?: ReactNode;
  title?: ReactNode;
};

export function UiSidebarPanel({
  actions,
  children,
  className,
  icon,
  subtitle,
  title,
  ...props
}: UiSidebarPanelProps) {
  const hasHeader = title !== undefined || subtitle !== undefined || icon !== undefined || actions !== undefined;

  return (
    <UiPanel className={['ui-sidebar-panel', className].filter(Boolean).join(' ')} {...props}>
      {hasHeader && (
        <header className="ui-sidebar-panel-header">
          {icon && <span className="ui-sidebar-panel-icon">{icon}</span>}
          <span className="ui-sidebar-panel-heading">
            {title !== undefined && <strong>{title}</strong>}
            {subtitle !== undefined && <small>{subtitle}</small>}
          </span>
          {actions && <span className="ui-sidebar-panel-actions">{actions}</span>}
        </header>
      )}
      <div className="ui-sidebar-panel-content">{children}</div>
    </UiPanel>
  );
}
