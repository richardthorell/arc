import type { CSSProperties, HTMLAttributes, PointerEventHandler, ReactNode } from 'react';

import './UiNodeCard.css';

export type UiNodeCardTone = 'default' | 'accent';

type UiNodeCardStyle = CSSProperties & {
  '--ui-node-card-color'?: string;
};

export type UiNodeCardProps = Omit<HTMLAttributes<HTMLElement>, 'children'> & {
  badge?: ReactNode;
  badgeTitle?: string;
  children: ReactNode;
  heading: ReactNode;
  icon?: ReactNode;
  nodeColor?: string;
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
  icon,
  nodeColor,
  onHeaderPointerDown,
  selected = false,
  style,
  tone = 'default',
  ...props
}: UiNodeCardProps) {
  const cardStyle: UiNodeCardStyle = { ...style };
  if (nodeColor) cardStyle['--ui-node-card-color'] = nodeColor;

  return (
    <article
      {...props}
      className={[
        'ui-node-card',
        `ui-node-card-${tone}`,
        nodeColor ? 'ui-node-card-has-color' : '',
        icon !== undefined && icon !== null ? 'ui-node-card-has-icon' : '',
        selected ? 'is-selected selected' : '',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      style={cardStyle}
    >
      <header className="ui-node-card-header" onPointerDown={onHeaderPointerDown}>
        {icon !== undefined && icon !== null && (
          <span aria-hidden="true" className="ui-node-card-icon">
            {icon}
          </span>
        )}
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
