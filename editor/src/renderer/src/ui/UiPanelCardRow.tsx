import type { HTMLAttributes, ReactNode } from 'react';

export type UiPanelCardRowProps = HTMLAttributes<HTMLDivElement> & {
  label: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  controlClassName?: string;
};

export function UiPanelCardRow({
  label,
  description,
  children,
  className,
  controlClassName,
  ...props
}: UiPanelCardRowProps) {
  return (
    <div className={['ui-panel-card-row', className].filter(Boolean).join(' ')} {...props}>
      <div className="ui-panel-card-row-label">
        <span>{label}</span>
        {description && <small>{description}</small>}
      </div>
      <div className={['ui-panel-card-row-control', controlClassName].filter(Boolean).join(' ')}>{children}</div>
    </div>
  );
}
