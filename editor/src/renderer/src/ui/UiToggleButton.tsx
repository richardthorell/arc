import type { ButtonHTMLAttributes, ReactNode } from 'react';

import './UiToggleButton.css';

export type UiToggleButtonProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  'aria-pressed' | 'onChange' | 'onClick' | 'role' | 'type'
> & {
  checked: boolean;
  mixed?: boolean;
  onCheckedChange: (checked: boolean) => void;
  label?: ReactNode;
  labelPosition?: 'start' | 'end';
};

export function UiToggleButton({
  checked,
  mixed = false,
  onCheckedChange,
  label,
  labelPosition = 'end',
  className,
  disabled,
  ...props
}: UiToggleButtonProps) {
  const labelNode = label ? <span className="ui-toggle-button-label">{label}</span> : null;
  return (
    <button
      {...props}
      aria-checked={mixed ? false : checked}
      data-mixed={mixed || undefined}
      className={[
        'ui-toggle-button',
        checked && !mixed ? 'is-checked' : '',
        mixed ? 'is-mixed' : '',
        label ? 'has-label' : '',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      disabled={disabled}
      role="switch"
      type="button"
      onClick={() => onCheckedChange(mixed ? true : !checked)}
    >
      {labelPosition === 'start' && labelNode}
      <span className="ui-toggle-button-track" aria-hidden="true">
        <span className="ui-toggle-button-thumb" />
      </span>
      {labelPosition === 'end' && labelNode}
    </button>
  );
}
