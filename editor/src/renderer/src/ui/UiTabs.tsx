import { X } from 'lucide-react';
import type { ButtonHTMLAttributes, HTMLAttributes, MouseEventHandler, ReactNode } from 'react';

import './UiTabs.css';

type UiTabsProps = HTMLAttributes<HTMLElement> & {
  children: ReactNode;
};

type UiTabProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
  children: ReactNode;
  icon?: ReactNode;
  closeLabel?: string;
  onClose?: MouseEventHandler<HTMLButtonElement>;
};

export function UiTabs({ children, className, ...props }: UiTabsProps) {
  return (
    <header className={['ui-tabs', className].filter(Boolean).join(' ')} {...props}>
      {children}
    </header>
  );
}

export function UiTab({
  active = false,
  children,
  className,
  icon,
  closeLabel,
  onClose,
  disabled,
  ...props
}: UiTabProps) {
  const classes = ['ui-tab', active ? 'is-active active' : '', className].filter(Boolean).join(' ');
  const label = typeof children === 'string' ? children : 'tab';
  const contents = (
    <>
      {icon && (
        <span aria-hidden="true" className="ui-tab-icon">
          {icon}
        </span>
      )}
      <span className="ui-tab-label">{children}</span>
    </>
  );

  if (!onClose) {
    return (
      <button className={classes} disabled={disabled} {...props}>
        {contents}
      </button>
    );
  }

  return (
    <span
      className={['ui-tab-shell', active ? 'is-active active' : '', disabled ? 'is-disabled' : '']
        .filter(Boolean)
        .join(' ')}
    >
      <button className={classes} disabled={disabled} {...props}>
        {contents}
      </button>
      <button
        aria-label={closeLabel ?? `Close ${label}`}
        className="ui-tab-close"
        disabled={disabled}
        onClick={(event) => {
          event.stopPropagation();
          onClose(event);
        }}
        type="button"
      >
        <X aria-hidden="true" size={12} />
      </button>
    </span>
  );
}
