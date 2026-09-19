import { forwardRef, type HTMLAttributes, type ReactNode } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

export type UiPanelCardProps = Omit<HTMLAttributes<HTMLElement>, 'title'> & {
  title: ReactNode;
  expandable?: boolean;
  collapsed?: boolean;
  onToggle?: () => void;
  actions?: ReactNode;
  children: ReactNode;
  contentClassName?: string;
};

export const UiPanelCard = forwardRef<HTMLElement, UiPanelCardProps>(function UiPanelCard(
  { title, expandable = true, collapsed = false, onToggle, actions, children, className, contentClassName, ...props },
  ref,
) {
  const isCollapsed = expandable && collapsed;
  const toggleLabel = typeof title === 'string' ? `${isCollapsed ? 'Expand' : 'Collapse'} ${title}` : undefined;

  return (
    <section
      className={['ui-panel-section', 'ui-panel-card', isCollapsed ? 'is-collapsed' : '', className]
        .filter(Boolean)
        .join(' ')}
      ref={ref}
      {...props}
    >
      <header className="ui-panel-section-header ui-panel-card-header">
        {expandable && onToggle ? (
          <button
            aria-expanded={!isCollapsed}
            aria-label={toggleLabel}
            className="ui-panel-section-toggle ui-panel-card-toggle"
            onClick={onToggle}
            type="button"
          >
            {isCollapsed ? <ChevronRight aria-hidden="true" size={15} /> : <ChevronDown aria-hidden="true" size={15} />}
            <span className="ui-panel-card-heading">{title}</span>
          </button>
        ) : (
          <div className="ui-panel-section-title ui-panel-card-title">
            <span className="ui-panel-card-heading">{title}</span>
          </div>
        )}
        {actions && <div className="ui-panel-section-actions ui-panel-card-actions">{actions}</div>}
      </header>
      {!isCollapsed && (
        <div
          className={['ui-panel-section-content', 'ui-panel-card-content', contentClassName].filter(Boolean).join(' ')}
        >
          {children}
        </div>
      )}
    </section>
  );
});
