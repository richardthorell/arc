import type { HTMLAttributes, PointerEventHandler, ReactNode } from 'react';

import './UiNodeCard.css';

export type UiNodeCardTone = 'default' | 'accent';

export type UiNodeCardProps = Omit<HTMLAttributes<HTMLElement>, 'children'> & {
  badge?: ReactNode;
  badgeTitle?: string;
  children: ReactNode;
  heading: ReactNode;
  onHeaderPointerDown?: PointerEventHandler<HTMLElement>;
  selected?: boolean;
  tone?: UiNodeCardTone;
};

export function UiNodeCard({
  badge,
  badgeTitle,
  children,
  className,
  heading,
  onHeaderPointerDown,
  selected = false,
  tone = 'default',
  ...props
}: UiNodeCardProps) {
  return (
    <article
      {...props}
      className={['ui-node-card', `ui-node-card-${tone}`, selected ? 'is-selected selected' : '', className]
        .filter(Boolean)
        .join(' ')}
    >
      <header className="ui-node-card-header" onPointerDown={onHeaderPointerDown}>
        <strong className="ui-node-card-title">{heading}</strong>
        {badge !== undefined && badge !== null && (
          <span className="ui-node-card-badge" title={badgeTitle}>
            {badge}
          </span>
        )}
      </header>
      {children}
    </article>
  );
}
