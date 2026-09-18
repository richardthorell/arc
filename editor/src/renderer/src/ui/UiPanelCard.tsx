import { forwardRef, type HTMLAttributes, type ReactNode } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

export type UiPanelCardProps = Omit<HTMLAttributes<HTMLElement>, 'title'> & {
  title: ReactNode;
  subtitle?: ReactNode;
  collapsed?: boolean;
  onToggle?: () => void;
  actions?: ReactNode;
  children: ReactNode;
  contentClassName?: string;
};

export const UiPanelCard = forwardRef<HTMLElement, UiPanelCardProps>(function UiPanelCard(
  { title, subtitle, collapsed = false, onToggle, actions, children, className, contentClassName, ...props },
  ref,
) {
  const toggleLabel = typeof title === 'string' ? `${collapsed ? 'Expand' : 'Collapse'} ${title}` : undefined;

  return (
    <section
      className={['ui-panel-section', 'ui-panel-card', collapsed ? 'is-collapsed' : '', className]
        .filter(Boolean)
        .join(' ')}
      ref={ref}
      {...props}
    >
      <header className="ui-panel-section-header ui-panel-card-header">
        {onToggle ? (
          <button
            aria-expanded={!collapsed}
            aria-label={toggleLabel}
            className="ui-panel-section-toggle ui-panel-card-toggle"
            onClick={onToggle}
            type="button"
          >
            {collapsed ? <ChevronRight aria-hidden="true" size={15} /> : <ChevronDown aria-hidden="true" size={15} />}
            <span className="ui-panel-card-heading">
              <span>{title}</span>
              {subtitle && <small className="ui-panel-card-subtitle">{subtitle}</small>}
            </span>
          </button>
        ) : (
          <div className="ui-panel-section-title ui-panel-card-title">
            <span className="ui-panel-card-heading">
              <span>{title}</span>
              {subtitle && <small className="ui-panel-card-subtitle">{subtitle}</small>}
            </span>
          </div>
        )}
        {actions && <div className="ui-panel-section-actions ui-panel-card-actions">{actions}</div>}
      </header>
      {!collapsed && (
        <div
          className={['ui-panel-section-content', 'ui-panel-card-content', contentClassName].filter(Boolean).join(' ')}
        >
          {children}
        </div>
      )}
    </section>
  );
});
