import type { HTMLAttributes, ReactNode } from 'react';

export type UiPanelCardRowProps = HTMLAttributes<HTMLDivElement> & {
  label?: ReactNode;
  description?: ReactNode;
  align?: 'center' | 'start';
  controlClassName?: string;
  children?: ReactNode;
  fullWidth?: boolean;
};

export function UiPanelCardRow({
  label,
  description,
  align = 'center',
  controlClassName,
  children,
  fullWidth = false,
  className,
  ...props
}: UiPanelCardRowProps) {
  return (
    <div
      className={[
        'ui-panel-card-row',
        align === 'start' ? 'is-start-aligned' : '',
        fullWidth ? 'is-full-width' : '',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      {...props}
    >
      {label !== undefined && (
        <span className="ui-panel-card-row-label">
          <span>{label}</span>
          {description && <small className="ui-panel-card-row-description">{description}</small>}
        </span>
      )}
      {children !== undefined && (
        <div className={['ui-panel-card-row-control', controlClassName].filter(Boolean).join(' ')}>{children}</div>
      )}
    </div>
  );
}
