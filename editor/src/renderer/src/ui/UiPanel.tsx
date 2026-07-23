import type { HTMLAttributes, ReactNode } from 'react';

type UiPanelProps = HTMLAttributes<HTMLElement> & {
  children: ReactNode;
};

type UiPanelHeaderProps = HTMLAttributes<HTMLElement> & {
  actions?: ReactNode;
  children: ReactNode;
};

export function UiPanel({ children, className, ...props }: UiPanelProps) {
  return (
    <section className={['ui-panel', className].filter(Boolean).join(' ')} {...props}>
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
