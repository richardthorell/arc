import type { HTMLAttributes, ReactNode } from 'react';

export type UiPanelCardRowProps = HTMLAttributes<HTMLDivElement> & {
  label: ReactNode;
  description?: ReactNode;
  align?: 'center' | 'start';
  controlClassName?: string;
  children: ReactNode;
};

export function UiPanelCardRow({
  label,
  description,
  align = 'center',
  controlClassName,
  children,
  className,
  ...props
}: UiPanelCardRowProps) {
  return (
    <div
      className={['ui-panel-card-row', align === 'start' ? 'is-start-aligned' : '', className]
        .filter(Boolean)
        .join(' ')}
      {...props}
    >
      <span className="ui-panel-card-row-label">
        <span>{label}</span>
        {description && <small className="ui-panel-card-row-description">{description}</small>}
      </span>
      <div className={['ui-panel-card-row-control', controlClassName].filter(Boolean).join(' ')}>{children}</div>
    </div>
  );
}
