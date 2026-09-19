import { Check, ChevronDown } from 'lucide-react';
import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';

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

type DropdownMenuPosition = {
  left: number;
  top: number;
  width: number;
  maxHeight: number;
};

const menuMargin = 8;
const menuGap = 4;
const minimumMenuWidth = 180;

export function UiDropdown<Value extends string = string>({
  value,
  options,
  onValueChange,
  ariaLabel,
  className,
  disabled = false,
}: UiDropdownProps<Value>) {
  const [open, setOpen] = useState(false);
  const [menuPosition, setMenuPosition] = useState<DropdownMenuPosition | null>(null);
  const rootRef = useRef<HTMLSpanElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const menuId = useId();
  const selected = options.find((option) => option.value === value) ?? options[0];

  const updateMenuPosition = useCallback(() => {
    const trigger = triggerRef.current;
    if (!trigger) return;

    const bounds = trigger.getBoundingClientRect();
    const width = Math.max(bounds.width, minimumMenuWidth);
    const left = Math.max(menuMargin, Math.min(bounds.right - width, window.innerWidth - width - menuMargin));
    const top = bounds.bottom + menuGap;

    setMenuPosition({
      left,
      top,
      width,
      maxHeight: Math.max(80, window.innerHeight - top - menuMargin),
    });
  }, []);

  useLayoutEffect(() => {
    if (!open) return;
    updateMenuPosition();
  }, [open, updateMenuPosition]);

  useEffect(() => {
    if (!open) return;

    const close = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!rootRef.current?.contains(target) && !menuRef.current?.contains(target)) setOpen(false);
    };
    const reposition = () => updateMenuPosition();

    window.addEventListener('pointerdown', close);
    window.addEventListener('resize', reposition);
    window.addEventListener('scroll', reposition, true);
    return () => {
      window.removeEventListener('pointerdown', close);
      window.removeEventListener('resize', reposition);
      window.removeEventListener('scroll', reposition, true);
    };
  }, [open, updateMenuPosition]);

  const choose = (option: UiDropdownOption<Value>) => {
    if (option.disabled) return;
    onValueChange(option.value);
    setOpen(false);
    triggerRef.current?.focus();
  };

  const menuStyle: CSSProperties | undefined = menuPosition
    ? {
        left: menuPosition.left,
        top: menuPosition.top,
        width: menuPosition.width,
        maxHeight: menuPosition.maxHeight,
      }
    : undefined;

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

      {open &&
        menuPosition &&
        createPortal(
          <div className="ui-dropdown-menu" id={menuId} ref={menuRef} role="listbox" style={menuStyle}>
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
          </div>,
          document.body,
        )}
    </span>
  );
}
