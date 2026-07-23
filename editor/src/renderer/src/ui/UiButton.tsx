import type { ButtonHTMLAttributes, ReactNode } from 'react';

type UiButtonVariant = 'default' | 'primary' | 'ghost' | 'icon' | 'toolbar' | 'danger';

type UiButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
  children: ReactNode;
  variant?: UiButtonVariant;
};

export function UiButton({ active = false, children, className, variant = 'default', ...props }: UiButtonProps) {
  const classes = ['ui-button', `ui-button-${variant}`, active ? 'is-active active' : '', className].filter(Boolean).join(' ');

  return (
    <button className={classes} {...props}>
      {children}
    </button>
  );
}
