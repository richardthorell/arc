import type { HTMLAttributes, ReactNode } from 'react';

import './UiSettingsCard.css';

export type UiSettingsCardProps = Omit<HTMLAttributes<HTMLElement>, 'title'> & {
  icon?: ReactNode;
  title: ReactNode;
  subtitle?: ReactNode;
  children: ReactNode;
};

export function UiSettingsCard({ icon, title, subtitle, children, className, ...props }: UiSettingsCardProps) {
  return (
    <section className={['ui-settings-card', className].filter(Boolean).join(' ')} {...props}>
      <header className="ui-settings-card-header">
        {icon && <span className="ui-settings-card-icon">{icon}</span>}
        <span className="ui-settings-card-heading">
          <strong>{title}</strong>
          {subtitle && <small>{subtitle}</small>}
        </span>
      </header>
      <div className="ui-settings-card-content">{children}</div>
    </section>
  );
}
