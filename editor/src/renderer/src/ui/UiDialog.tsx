import { useEffect, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react';
import { X } from 'lucide-react';

import { UiIconButton } from './UiIconButton';

import './UiDialog.css';

type UiDialogProps = {
  children?: ReactNode;
  className?: string;
  footer?: ReactNode;
  icon?: ReactNode;
  onClose?: () => void;
  preview?: boolean;
  subtitle?: string;
  title?: string;
  width?: number;
};

type DialogPosition = { x: number; y: number };
type DialogDrag = {
  pointer: { x: number; y: number };
  position: DialogPosition;
};

const interactiveDragTarget = (target: EventTarget | null): boolean =>
  target instanceof Element && Boolean(target.closest('button, a, input, textarea, select'));

export function UiDialog({
  children,
  className,
  footer,
  icon,
  onClose,
  preview = false,
  subtitle,
  title,
  width = 520,
}: UiDialogProps) {
  const [position, setPosition] = useState<DialogPosition>({ x: 0, y: 0 });
  const [drag, setDrag] = useState<DialogDrag | null>(null);
  const classes = ['ui-dialog-backdrop', preview ? 'is-preview' : ''].filter(Boolean).join(' ');
  const dialogClasses = ['ui-dialog', drag ? 'is-dragging' : '', className].filter(Boolean).join(' ');
  const draggable = !preview && Boolean(title || subtitle || icon || onClose);

  useEffect(() => {
    if (!drag) return;

    const move = (event: PointerEvent) => {
      setPosition({
        x: drag.position.x + event.clientX - drag.pointer.x,
        y: drag.position.y + event.clientY - drag.pointer.y,
      });
    };
    const up = () => setDrag(null);

    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up, { once: true });
    window.addEventListener('pointercancel', up, { once: true });
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
    };
  }, [drag]);

  const startDrag = (event: ReactPointerEvent<HTMLElement>) => {
    if (!draggable || event.button !== 0 || interactiveDragTarget(event.target)) return;
    event.preventDefault();
    setDrag({ pointer: { x: event.clientX, y: event.clientY }, position });
  };

  return (
    <div
      className={classes}
      onPointerDown={(event) => {
        if (!preview && onClose && event.target === event.currentTarget) onClose();
      }}
    >
      <section
        aria-label={title || 'Dialog'}
        aria-modal={preview ? undefined : true}
        className={dialogClasses}
        role="dialog"
        style={{ width, transform: preview ? undefined : `translate3d(${position.x}px, ${position.y}px, 0)` }}
      >
        {(title || subtitle || icon || onClose) && (
          <header
            className={draggable ? 'ui-dialog-header is-draggable' : 'ui-dialog-header'}
            onPointerDown={startDrag}
          >
            <div className="ui-dialog-heading">
              {icon}
              <span>
                {title && <strong>{title}</strong>}
                {subtitle && <small>{subtitle}</small>}
              </span>
            </div>
            {onClose && (
              <UiIconButton aria-label="Close dialog" label="Close dialog" onClick={onClose}>
                <X size={15} />
              </UiIconButton>
            )}
          </header>
        )}
        <div className="ui-dialog-body">{children}</div>
        {footer && <footer className="ui-dialog-footer">{footer}</footer>}
      </section>
    </div>
  );
}
