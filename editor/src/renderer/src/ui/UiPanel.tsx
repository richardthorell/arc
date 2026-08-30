import type { HTMLAttributes, ReactNode } from 'react';

import './UiPanel.css';

type UiPanelProps = HTMLAttributes<HTMLElement> & {
  children: ReactNode;
  variant?: 'default' | 'inspector';
};

type UiPanelHeaderProps = HTMLAttributes<HTMLElement> & {
  actions?: ReactNode;
  children: ReactNode;
};

export function UiPanel({ children, className, variant = 'default', ...props }: UiPanelProps) {
  return (
    <section className={['ui-panel', `ui-panel-${variant}`, className].filter(Boolean).join(' ')} {...props}>
      {children}
    </section>
  );
}

export function UiPanelHeader({ actions, children, className, ...props }: UiPanelHeaderProps) {
  return (
    <header className={['ui-panel-header', className].filter(Boolean).join(' ')} {...props}>
      <span>{children}</span>
      {actions}
    </header>
  );
}
