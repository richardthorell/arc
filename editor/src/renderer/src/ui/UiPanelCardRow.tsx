import type { HTMLAttributes, ReactNode } from 'react';

type UiPanelCardRowProps = HTMLAttributes<HTMLDivElement> & {
  label: ReactNode;
  children: ReactNode;
};

export function UiPanelCardRow({ label, children, className, ...props }: UiPanelCardRowProps) {
  return (
    <div className={['ui-panel-card-row', className].filter(Boolean).join(' ')} {...props}>
      <span className="ui-panel-card-row-label">{label}</span>
      <div className="ui-panel-card-row-control">{children}</div>
    </div>
  );
}
