import { forwardRef, type HTMLAttributes, type ReactNode } from 'react';

import './UiAgentCard.css';

export type UiAgentCardState = 'complete' | 'streaming' | 'error';

export type UiAgentCardProps = Omit<HTMLAttributes<HTMLElement>, 'title'> & {
  title?: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  actions?: ReactNode;
  state?: UiAgentCardState;
  children: ReactNode;
};

export const UiAgentCard = forwardRef<HTMLElement, UiAgentCardProps>(function UiAgentCard(
  { title, subtitle, icon, actions, state = 'complete', children, className, ...props },
  ref,
) {
  const hasHeader = Boolean(title || subtitle || icon || actions);

  return (
    <article
      className={['ui-agent-card', className].filter(Boolean).join(' ')}
      data-state={state}
      ref={ref}
      {...props}
    >
      {hasHeader && (
        <header className="ui-agent-card-header">
          {icon && <span className="ui-agent-card-icon">{icon}</span>}
          {(title || subtitle) && (
            <span className="ui-agent-card-heading">
              {title && <strong>{title}</strong>}
              {subtitle && <small>{subtitle}</small>}
            </span>
          )}
          {actions && <span className="ui-agent-card-actions">{actions}</span>}
        </header>
      )}
      <div className="ui-agent-card-content">{children}</div>
    </article>
  );
});

export type UiAgentTextCardProps = Omit<UiAgentCardProps, 'children'> & {
  text: string;
};

export function UiAgentTextCard({ text, state = 'complete', ...props }: UiAgentTextCardProps) {
  return (
    <UiAgentCard state={state} {...props}>
      <div className="ui-agent-text-card-content">{text || (state === 'streaming' ? '…' : '')}</div>
    </UiAgentCard>
  );
}
