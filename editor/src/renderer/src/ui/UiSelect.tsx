import { Check, ChevronDown } from 'lucide-react';
import { useEffect, useId, useRef, useState } from 'react';

import { UiButton } from './UiButton';

export type UiSelectOption = {
  value: string;
  label: string;
  disabled?: boolean;
};

type UiSelectProps = {
  value: string;
  options: ReadonlyArray<UiSelectOption>;
  onValueChange: (value: string) => void;
  ariaLabel?: string;
  className?: string;
  disabled?: boolean;
};

export function UiSelect({
  value,
  options,
  onValueChange,
  ariaLabel,
  className,
  disabled = false,
}: UiSelectProps) {
  const [open, setOpen] = useState(false);
  const selectedIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  );
  const [activeIndex, setActiveIndex] = useState(selectedIndex);
  const rootRef = useRef<HTMLSpanElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const listboxId = useId();
  const selected = options.find((option) => option.value === value) ?? options[0];

  useEffect(() => {
    if (!open) return;

    const close = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };

    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [open]);

  useEffect(() => {
    if (open) setActiveIndex(selectedIndex);
  }, [open, selectedIndex]);

  const moveActive = (direction: 1 | -1) => {
    if (!options.length) return;
    let next = activeIndex;
    for (let offset = 0; offset < options.length; offset += 1) {
      next = (next + direction + options.length) % options.length;
      if (!options[next]?.disabled) {
        setActiveIndex(next);
        return;
      }
    }
  };

  const choose = (nextValue: string) => {
    onValueChange(nextValue);
    setOpen(false);
    triggerRef.current?.focus();
  };

  return (
    <span ref={rootRef} className={['ui-select', className].filter(Boolean).join(' ')}>
      <UiButton
        ref={triggerRef}
        aria-controls={listboxId}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={ariaLabel}
        className="ui-select-trigger"
        disabled={disabled}
        role="combobox"
        type="button"
        onClick={() => setOpen((current) => !current)}
        onKeyDown={(event) => {
          if (event.key === 'ArrowDown') {
            event.preventDefault();
            if (!open) setOpen(true);
            else moveActive(1);
          } else if (event.key === 'ArrowUp') {
            event.preventDefault();
            if (!open) setOpen(true);
            else moveActive(-1);
          } else if (event.key === 'Home' && open) {
            event.preventDefault();
            const first = options.findIndex((option) => !option.disabled);
            if (first >= 0) setActiveIndex(first);
          } else if (event.key === 'End' && open) {
            event.preventDefault();
            for (let index = options.length - 1; index >= 0; index -= 1) {
              if (!options[index]?.disabled) {
                setActiveIndex(index);
                break;
              }
            }
          } else if ((event.key === 'Enter' || event.key === ' ') && open) {
            event.preventDefault();
            const option = options[activeIndex];
            if (option && !option.disabled) choose(option.value);
          } else if (event.key === 'Escape' && open) {
            event.preventDefault();
            setOpen(false);
          }
        }}
      >
        <span className="ui-select-value">{selected?.label ?? ''}</span>
        <ChevronDown aria-hidden="true" className="ui-select-chevron" size={12} />
      </UiButton>

      {open && (
        <div className="menu-dropdown ui-select-menu" id={listboxId} role="listbox">
          {options.map((option, index) => {
            const isSelected = option.value === value;
            return (
              <UiButton
                aria-selected={isSelected}
                className={index === activeIndex ? 'is-active-option' : undefined}
                disabled={option.disabled}
                key={option.value}
                role="option"
                type="button"
                variant="ghost"
                onMouseEnter={() => setActiveIndex(index)}
                onClick={() => choose(option.value)}
              >
                <span>{option.label}</span>
                <b className="menu-checkmark" aria-hidden="true">
                  {isSelected ? <Check size={13} /> : null}
                </b>
              </UiButton>
            );
          })}
        </div>
      )}
    </span>
  );
}
