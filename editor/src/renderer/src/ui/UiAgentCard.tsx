import { Check, Copy } from 'lucide-react';
import { forwardRef, useState, type HTMLAttributes, type ReactNode } from 'react';

import './UiAgentCard.css';

export type UiAgentCardState = 'complete' | 'streaming' | 'error';
export type UiAgentCardSide = 'left' | 'right' | 'none';
export type UiAgentCardTone = 'agent' | 'user' | 'neutral';
export type UiAgentCardActionAlignment = 'left' | 'right';

export type UiAgentCardActionRowProps = {
  children: ReactNode;
  align?: UiAgentCardActionAlignment;
  revealOnHover?: boolean;
  className?: string;
};

export function UiAgentCardActionRow({
  children,
  align = 'left',
  revealOnHover = false,
  className,
}: UiAgentCardActionRowProps) {
  return (
    <div
      className={['ui-agent-card-action-row', className].filter(Boolean).join(' ')}
      data-align={align}
      data-reveal-on-hover={revealOnHover ? 'true' : 'false'}
    >
      {children}
    </div>
  );
}

export function UiAgentCardCopyAction({ value, label = 'Copy' }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    if (!navigator.clipboard?.writeText) return;
    await navigator.clipboard.writeText(value);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };

  return (
    <button
      aria-label={copied ? 'Copied' : label}
      className="ui-agent-card-action-button"
      title={copied ? 'Copied' : label}
      type="button"
      onClick={() => void copy()}
    >
      {copied ? <Check aria-hidden="true" size={15} /> : <Copy aria-hidden="true" size={15} />}
    </button>
  );
}

export type UiAgentCardProps = Omit<HTMLAttributes<HTMLElement>, 'title'> & {
  title?: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  actions?: ReactNode;
  footerActions?: ReactNode;
  timestamp?: ReactNode;
  state?: UiAgentCardState;
  side?: UiAgentCardSide;
  tone?: UiAgentCardTone;
  children: ReactNode;
};

export const UiAgentCard = forwardRef<HTMLElement, UiAgentCardProps>(function UiAgentCard(
  {
    title,
    subtitle,
    icon,
    actions,
    footerActions,
    timestamp,
    state = 'complete',
    side = 'none',
    tone = 'neutral',
    children,
    className,
    ...props
  },
  ref,
) {
  const hasHeader = Boolean(title || subtitle || icon || actions);

  return (
    <article
      className={['ui-agent-card', className].filter(Boolean).join(' ')}
      data-has-footer-actions={footerActions ? 'true' : 'false'}
      data-side={side}
      data-state={state}
      data-tone={tone}
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
      {timestamp && <footer className="ui-agent-card-timestamp">{timestamp}</footer>}
      {footerActions && <footer className="ui-agent-card-footer-actions">{footerActions}</footer>}
    </article>
  );
});

export type UiAgentTextCardProps = Omit<UiAgentCardProps, 'children'> & {
  text: string;
  renderText?: (text: string) => ReactNode;
  streamingPlaceholder?: ReactNode;
};

export function UiAgentTextCard({
  text,
  renderText,
  state = 'complete',
  side = 'none',
  tone = 'neutral',
  footerActions,
  streamingPlaceholder = '…',
  ...props
}: UiAgentTextCardProps) {
  const content = text ? (renderText ? renderText(text) : text) : state === 'streaming' ? streamingPlaceholder : '';
  const actionRow = (
    <UiAgentCardActionRow align={side === 'right' ? 'right' : 'left'} revealOnHover={tone === 'user'}>
      <UiAgentCardCopyAction label="Copy message" value={text} />
      {footerActions}
    </UiAgentCardActionRow>
  );
  return (
    <UiAgentCard footerActions={actionRow} side={side} state={state} tone={tone} {...props}>
      <div className="ui-agent-text-card-content">{content}</div>
    </UiAgentCard>
  );
}
