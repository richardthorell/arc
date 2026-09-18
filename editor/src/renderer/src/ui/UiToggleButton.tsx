import type { ButtonHTMLAttributes } from 'react';

export type UiToggleButtonProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  'aria-pressed' | 'onChange' | 'onClick' | 'role' | 'type'
> & {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
};

export function UiToggleButton({ checked, onCheckedChange, className, disabled, ...props }: UiToggleButtonProps) {
  return (
    <button
      {...props}
      aria-checked={checked}
      className={['ui-toggle-button', checked ? 'is-checked' : '', className].filter(Boolean).join(' ')}
      data-state={checked ? 'checked' : 'unchecked'}
      disabled={disabled}
      role="switch"
      type="button"
      onClick={() => onCheckedChange(!checked)}
    >
      <span className="ui-toggle-button-track" aria-hidden="true">
        <span className="ui-toggle-button-thumb" />
      </span>
    </button>
  );
}
