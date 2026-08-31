import { forwardRef, type HTMLAttributes, type ReactNode } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

type UiPanelSectionProps = Omit<HTMLAttributes<HTMLElement>, 'title'> & {
  title: ReactNode;
  collapsed?: boolean;
  onToggle?: () => void;
  actions?: ReactNode;
  children: ReactNode;
  contentClassName?: string;
};

export const UiPanelSection = forwardRef<HTMLElement, UiPanelSectionProps>(function UiPanelSection(
  { title, collapsed = false, onToggle, actions, children, className, contentClassName, ...props },
  ref,
) {
  return (
    <section
      className={['ui-panel-section', collapsed ? 'is-collapsed' : '', className].filter(Boolean).join(' ')}
      ref={ref}
      {...props}
    >
      <header className="ui-panel-section-header">
        {onToggle ? (
          <button aria-expanded={!collapsed} className="ui-panel-section-toggle" onClick={onToggle} type="button">
            {collapsed ? <ChevronRight aria-hidden="true" size={15} /> : <ChevronDown aria-hidden="true" size={15} />}
            <span>{title}</span>
          </button>
        ) : (
          <div className="ui-panel-section-title">{title}</div>
        )}
        {actions && <div className="ui-panel-section-actions">{actions}</div>}
      </header>
      {!collapsed && (
        <div className={['ui-panel-section-content', contentClassName].filter(Boolean).join(' ')}>{children}</div>
      )}
    </section>
  );
});
