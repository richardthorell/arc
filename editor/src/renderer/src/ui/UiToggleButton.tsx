import type { ButtonHTMLAttributes } from 'react';

type UiToggleButtonProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'aria-pressed' | 'onChange'> & {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
};

export function UiToggleButton({
  checked,
  onCheckedChange,
  className,
  disabled,
  ...props
}: UiToggleButtonProps) {
  return (
    <button
      aria-checked={checked}
      className={['ui-toggle-button', checked ? 'is-checked' : '', className].filter(Boolean).join(' ')}
      disabled={disabled}
      role="switch"
      type="button"
      onClick={() => onCheckedChange(!checked)}
      {...props}
    >
      <span className="ui-toggle-button-track" aria-hidden="true">
        <span className="ui-toggle-button-thumb" />
      </span>
    </button>
  );
}
