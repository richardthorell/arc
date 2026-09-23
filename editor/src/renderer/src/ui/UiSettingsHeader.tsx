import type { HTMLAttributes, ReactNode } from 'react';

import './UiSettingsHeader.css';

const defaultBackground = (
  <svg
    aria-hidden="true"
    className="ui-settings-header-default-artwork"
    preserveAspectRatio="xMidYMid slice"
    viewBox="0 0 960 180"
  >
    <path d="M570 188C620 93 718 31 832 23C884 19 931 27 974 46" />
    <path d="M645 198C686 121 765 70 857 64C901 61 941 67 978 84" />
    <path d="M725 194C754 143 807 111 869 107C909 104 945 110 977 124" />
    <circle cx="847" cy="24" r="132" />
    <circle cx="847" cy="24" r="92" />
  </svg>
);

export type UiSettingsHeaderProps = Omit<HTMLAttributes<HTMLElement>, 'title'> & {
  title: ReactNode;
  subtitle?: ReactNode;
  background?: ReactNode;
};

export function UiSettingsHeader({ title, subtitle, background, className, ...props }: UiSettingsHeaderProps) {
  return (
    <header className={['ui-settings-header', className].filter(Boolean).join(' ')} {...props}>
      <div aria-hidden="true" className="ui-settings-header-background">
        {background ?? defaultBackground}
      </div>
      <div className="ui-settings-header-copy">
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </div>
    </header>
  );
}
