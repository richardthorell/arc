import { ChevronDown } from 'lucide-react';
import { useEffect, useId, useRef, useState, type ReactNode } from 'react';

import { UiButton, type UiButtonVariant } from './UiButton';

import './UiSplitButton.css';

export type UiSplitButtonOption<Value extends string = string> = {
  value: Value;
  label: string;
  icon?: ReactNode;
  disabled?: boolean;
};

type UiSplitButtonProps<Value extends string = string> = {
  label: string;
  icon?: ReactNode;
  options: ReadonlyArray<UiSplitButtonOption<Value>>;
  onClick: () => void;
  onOptionSelect: (value: Value) => void;
  ariaLabel?: string;
  menuAriaLabel?: string;
  className?: string;
  disabled?: boolean;
  variant?: UiButtonVariant;
};

export function UiSplitButton<Value extends string = string>({
  label,
  icon,
  options,
  onClick,
  onOptionSelect,
  ariaLabel,
  menuAriaLabel = `${label} actions`,
  className,
  disabled = false,
  variant = 'primary',
}: UiSplitButtonProps<Value>) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLSpanElement | null>(null);
  const menuTriggerRef = useRef<HTMLButtonElement | null>(null);
  const menuId = useId();

  useEffect(() => {
    if (!open) return;

    const close = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };

    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [open]);

  const choose = (option: UiSplitButtonOption<Value>) => {
    if (option.disabled) return;
    onOptionSelect(option.value);
    setOpen(false);
    menuTriggerRef.current?.focus();
  };

  return (
    <span ref={rootRef} className={['ui-split-button', className].filter(Boolean).join(' ')}>
      <UiButton
        aria-label={ariaLabel}
        className="ui-split-button-main"
        disabled={disabled}
        type="button"
        variant={variant}
        onClick={onClick}
      >
        {icon && <span className="ui-split-button-icon">{icon}</span>}
        <span>{label}</span>
      </UiButton>
      <UiButton
        ref={menuTriggerRef}
        aria-controls={menuId}
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label={menuAriaLabel}
        className="ui-split-button-menu-trigger"
        disabled={disabled}
        type="button"
        variant={variant}
        onClick={() => setOpen((current) => !current)}
        onKeyDown={(event) => {
          if ((event.key === 'ArrowDown' || event.key === 'Enter' || event.key === ' ') && !open) {
            event.preventDefault();
            setOpen(true);
          } else if (event.key === 'Escape' && open) {
            event.preventDefault();
            setOpen(false);
            menuTriggerRef.current?.focus();
          }
        }}
      >
        <ChevronDown aria-hidden="true" size={12} />
      </UiButton>

      {open && (
        <div className="ui-split-button-menu" id={menuId} role="menu">
          {options.map((option) => (
            <UiButton
              className="ui-split-button-option"
              disabled={option.disabled}
              key={option.value}
              role="menuitem"
              type="button"
              variant="ghost"
              onClick={() => choose(option)}
            >
              {option.icon && <span className="ui-split-button-icon">{option.icon}</span>}
              <span>{option.label}</span>
            </UiButton>
          ))}
        </div>
      )}
    </span>
  );
}
