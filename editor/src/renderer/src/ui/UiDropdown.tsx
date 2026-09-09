import { Check, ChevronDown } from 'lucide-react';
import { useEffect, useId, useRef, useState, type ReactNode } from 'react';

import { UiButton } from './UiButton';

import './UiDropdown.css';

export type UiDropdownOption<Value extends string = string> = {
  value: Value;
  label: string;
  icon?: ReactNode;
  disabled?: boolean;
};

type UiDropdownProps<Value extends string = string> = {
  value: Value;
  options: ReadonlyArray<UiDropdownOption<Value>>;
  onValueChange: (value: Value) => void;
  ariaLabel?: string;
  className?: string;
  disabled?: boolean;
};

export function UiDropdown<Value extends string = string>({
  value,
  options,
  onValueChange,
  ariaLabel,
  className,
  disabled = false,
}: UiDropdownProps<Value>) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLSpanElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const menuId = useId();
  const selected = options.find((option) => option.value === value) ?? options[0];

  useEffect(() => {
    if (!open) return;

    const close = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };

    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [open]);

  const choose = (option: UiDropdownOption<Value>) => {
    if (option.disabled) return;
    onValueChange(option.value);
    setOpen(false);
    triggerRef.current?.focus();
  };

  return (
    <span ref={rootRef} className={['ui-dropdown', className].filter(Boolean).join(' ')}>
      <UiButton
        ref={triggerRef}
        aria-controls={menuId}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={ariaLabel}
        className="ui-dropdown-trigger"
        disabled={disabled}
        type="button"
        variant="toolbar"
        onClick={() => setOpen((current) => !current)}
        onKeyDown={(event) => {
          if ((event.key === 'ArrowDown' || event.key === 'Enter' || event.key === ' ') && !open) {
            event.preventDefault();
            setOpen(true);
          } else if (event.key === 'Escape' && open) {
            event.preventDefault();
            setOpen(false);
            triggerRef.current?.focus();
          }
        }}
      >
        <span className="ui-dropdown-value">
          {selected?.icon && <span className="ui-dropdown-icon">{selected.icon}</span>}
          <span>{selected?.label ?? ''}</span>
        </span>
        <ChevronDown aria-hidden="true" className="ui-dropdown-chevron" size={12} />
      </UiButton>

      {open && (
        <div className="ui-dropdown-menu" id={menuId} role="listbox">
          {options.map((option) => {
            const isSelected = option.value === value;
            return (
              <UiButton
                aria-selected={isSelected}
                className="ui-dropdown-option"
                disabled={option.disabled}
                key={option.value}
                role="option"
                type="button"
                variant="ghost"
                onClick={() => choose(option)}
              >
                <span className="ui-dropdown-option-content">
                  {option.icon && <span className="ui-dropdown-icon">{option.icon}</span>}
                  <span>{option.label}</span>
                </span>
                <span className="ui-dropdown-check" aria-hidden="true">
                  {isSelected ? <Check size={13} /> : null}
                </span>
              </UiButton>
            );
          })}
        </div>
      )}
    </span>
  );
}
