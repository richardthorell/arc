import type { ButtonHTMLAttributes, ReactNode } from 'react';

import { UiButton } from './UiButton';

type UiIconButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> & {
  active?: boolean;
  children: ReactNode;
  label: string;
};

export function UiIconButton({ active = false, children, className, label, title, ...props }: UiIconButtonProps) {
  return (
    <UiButton
      active={active}
      aria-label={label}
      className={['ui-icon-button', className].filter(Boolean).join(' ')}
      title={title ?? label}
      variant="icon"
      {...props}
    >
      {children}
    </UiButton>
  );
}
