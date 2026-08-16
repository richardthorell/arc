import { forwardRef } from 'react';
import type { ButtonHTMLAttributes, ReactNode } from 'react';

import { UiButton } from './UiButton';
import type { UiButtonVariant } from './UiButton';

type UiIconButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> & {
  active?: boolean;
  children: ReactNode;
  label: string;
  variant?: UiButtonVariant;
};

export const UiIconButton = forwardRef<HTMLButtonElement, UiIconButtonProps>(function UiIconButton(
  { active = false, children, className, label, title, variant = 'icon', ...props },
  ref,
) {
  return (
    <UiButton
      active={active}
      aria-label={label}
      className={['ui-icon-button', className].filter(Boolean).join(' ')}
      ref={ref}
      title={title ?? label}
      variant={variant}
      {...props}
    >
      {children}
    </UiButton>
  );
});
