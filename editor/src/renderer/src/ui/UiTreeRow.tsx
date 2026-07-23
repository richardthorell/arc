import type { ButtonHTMLAttributes, CSSProperties, HTMLAttributes, ReactNode } from 'react';

type UiTreeRowCommonProps = {
  depth?: number;
  meta?: ReactNode;
  selected?: boolean;
  children: ReactNode;
};
type UiTreeRowProps = UiTreeRowCommonProps & (
  | ({ as?: 'button' } & ButtonHTMLAttributes<HTMLButtonElement>)
  | ({ as: 'div' } & HTMLAttributes<HTMLDivElement>)
);

type TreeRowStyle = CSSProperties & {
  '--ui-tree-depth'?: string;
};

export function UiTreeRow({ as = 'button', children, className, depth = 0, meta, selected = false, style, ...props }: UiTreeRowProps) {
  const classes = ['ui-tree-row', selected ? 'is-selected selected' : '', className].filter(Boolean).join(' ');
  const rowStyle: TreeRowStyle = {
    ...style,
    '--ui-tree-depth': `${depth * 14}px`,
  };

  const contents = <>{children}{meta}</>;
  if (as === 'div') {
    return <div className={classes} style={rowStyle} {...props as HTMLAttributes<HTMLDivElement>}>{contents}</div>;
  }
  return (
    <button className={classes} style={rowStyle} type="button" {...props as ButtonHTMLAttributes<HTMLButtonElement>}>
      {contents}
    </button>
  );
}
