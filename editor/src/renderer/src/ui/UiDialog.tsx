import { useEffect, useState, type CSSProperties, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react';
import { X } from 'lucide-react';

import { UiIconButton } from './UiIconButton';

import './UiDialog.css';

export type UiDialogPosition = { x: number; y: number };

export type UiDialogProps = {
  ariaLabel?: string;
  blurBackdrop?: boolean;
  children?: ReactNode;
  className?: string;
  draggable?: boolean;
  footer?: ReactNode;
  icon?: ReactNode;
  initialPosition?: UiDialogPosition;
  modal?: boolean;
  onClose?: () => void;
  preview?: boolean;
  subtitle?: string;
  title?: string;
  width?: number;
  zIndex?: CSSProperties['zIndex'];
};

type DialogDrag = {
  pointer: { x: number; y: number };
  position: UiDialogPosition;
};

const interactiveDragTarget = (target: EventTarget | null): boolean =>
  target instanceof Element && Boolean(target.closest('button, a, input, textarea, select'));

export function UiDialog({
  ariaLabel,
  blurBackdrop = true,
  children,
  className,
  draggable: draggableProp,
  footer,
  icon,
  initialPosition,
  modal = true,
  onClose,
  preview = false,
  subtitle,
  title,
  width = 520,
  zIndex,
}: UiDialogProps) {
  const [position, setPosition] = useState<UiDialogPosition>(() => initialPosition ?? { x: 0, y: 0 });
  const [drag, setDrag] = useState<DialogDrag | null>(null);
  const modeless = !modal && !preview;
  const classes = [
    'ui-dialog-backdrop',
    preview ? 'is-preview' : '',
    modeless ? 'is-modeless' : '',
    !blurBackdrop ? 'is-no-blur' : '',
  ]
    .filter(Boolean)
    .join(' ');
  const dialogClasses = ['ui-dialog', drag ? 'is-dragging' : '', className].filter(Boolean).join(' ');
  const hasHeader = Boolean(title || subtitle || icon || onClose);
  const draggable = draggableProp ?? (!preview && hasHeader);

  useEffect(() => {
    if (!initialPosition) return;
    setPosition(initialPosition);
  }, [initialPosition?.x, initialPosition?.y]);

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

  const dialogStyle: CSSProperties = modeless
    ? { width, left: position.x, top: position.y }
    : { width, transform: preview ? undefined : `translate3d(${position.x}px, ${position.y}px, 0)` };

  return (
    <div
      className={classes}
      style={zIndex !== undefined ? { zIndex } : undefined}
      onPointerDown={(event) => {
        if (!preview && modal && onClose && event.target === event.currentTarget) onClose();
      }}
    >
      <section
        aria-label={ariaLabel ?? title ?? 'Dialog'}
        aria-modal={!preview && modal ? true : undefined}
        className={dialogClasses}
        role="dialog"
        style={dialogStyle}
      >
        {hasHeader && (
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
