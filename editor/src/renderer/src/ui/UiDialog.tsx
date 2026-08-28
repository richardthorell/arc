import type { ReactNode } from 'react';
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
  const classes = ['ui-dialog-backdrop', preview ? 'is-preview' : ''].filter(Boolean).join(' ');
  const dialogClasses = ['ui-dialog', className].filter(Boolean).join(' ');

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
        style={{ width }}
      >
        {(title || subtitle || icon || onClose) && (
          <header className="ui-dialog-header">
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
