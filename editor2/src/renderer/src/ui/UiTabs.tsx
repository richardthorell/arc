import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from 'react';

type UiTabsProps = HTMLAttributes<HTMLElement> & {
  children: ReactNode;
};

type UiTabProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
  children: ReactNode;
};

export function UiTabs({ children, className, ...props }: UiTabsProps) {
  return (
    <header className={['ui-tabs', className].filter(Boolean).join(' ')} {...props}>
      {children}
    </header>
  );
}

export function UiTab({ active = false, children, className, ...props }: UiTabProps) {
  const classes = ['ui-tab', active ? 'is-active active' : '', className].filter(Boolean).join(' ');

  return (
    <button className={classes} {...props}>
      {children}
    </button>
  );
}
