import type { ButtonHTMLAttributes, ReactNode } from 'react';

export type UiToggleButtonProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  'aria-pressed' | 'onChange' | 'onClick' | 'role' | 'type'
> & {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  label?: ReactNode;
  labelPosition?: 'start' | 'end';
};

export function UiToggleButton({
  checked,
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
      aria-checked={checked}
      className={['ui-toggle-button', checked ? 'is-checked' : '', label ? 'has-label' : '', className]
        .filter(Boolean)
        .join(' ')}
      disabled={disabled}
      role="switch"
      type="button"
      onClick={() => onCheckedChange(!checked)}
    >
      {labelPosition === 'start' && labelNode}
      <span className="ui-toggle-button-track" aria-hidden="true">
        <span className="ui-toggle-button-thumb" />
      </span>
      {labelPosition === 'end' && labelNode}
    </button>
  );
}
