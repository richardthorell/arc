import type { SelectHTMLAttributes } from 'react';
import { ChevronDown } from 'lucide-react';

type UiSelectProps = SelectHTMLAttributes<HTMLSelectElement>;

export function UiSelect({ className, children, ...props }: UiSelectProps) {
  return (
    <span className="ui-select">
      <select className={['ui-select-input', className].filter(Boolean).join(' ')} {...props}>
        {children}
      </select>
      <ChevronDown aria-hidden="true" className="ui-select-chevron" size={12} />
    </span>
  );
}
