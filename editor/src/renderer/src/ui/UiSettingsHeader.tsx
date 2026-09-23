import type { HTMLAttributes, ReactNode } from 'react';

import './UiSettingsHeader.css';

export type UiSettingsHeaderProps = Omit<HTMLAttributes<HTMLElement>, 'title'> & {
  title: ReactNode;
  subtitle?: ReactNode;
};

export function UiSettingsHeader({ title, subtitle, className, ...props }: UiSettingsHeaderProps) {
  return (
    <header className={['ui-settings-header', className].filter(Boolean).join(' ')} {...props}>
      <h2>{title}</h2>
      {subtitle && <p>{subtitle}</p>}
    </header>
  );
}
