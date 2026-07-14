import type { ButtonHTMLAttributes, CSSProperties, ReactNode } from 'react';

type UiTreeRowProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  depth?: number;
  meta?: ReactNode;
  selected?: boolean;
  children: ReactNode;
};

type TreeRowStyle = CSSProperties & {
  '--ui-tree-depth'?: string;
};

export function UiTreeRow({ children, className, depth = 0, meta, selected = false, style, ...props }: UiTreeRowProps) {
  const classes = ['ui-tree-row', selected ? 'is-selected selected' : '', className].filter(Boolean).join(' ');
  const rowStyle: TreeRowStyle = {
    ...style,
    '--ui-tree-depth': `${depth * 14}px`,
  };

  return (
    <button className={classes} style={rowStyle} {...props}>
      {children}
      {meta}
    </button>
  );
}
