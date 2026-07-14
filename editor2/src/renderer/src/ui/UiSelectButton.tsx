import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';

import { UiButton } from './UiButton';

type UiSelectButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
  children: ReactNode;
  showChevron?: boolean;
};

export function UiSelectButton({ active = false, children, className, showChevron = true, ...props }: UiSelectButtonProps) {
  return (
    <UiButton
      active={active}
      className={['ui-select-button', className].filter(Boolean).join(' ')}
      variant="toolbar"
      {...props}
    >
      <span className="ui-select-button-label">{children}</span>
      {showChevron && <ChevronDown className="ui-select-button-chevron" size={12} />}
    </UiButton>
  );
}
