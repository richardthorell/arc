import { forwardRef } from 'react';
import type { HTMLAttributes, ReactNode } from 'react';

import '../layout/toolbar.css';

export type UiEditorToolbarProps = Omit<HTMLAttributes<HTMLDivElement>, 'children'> & {
  left?: ReactNode;
  center?: ReactNode;
  right?: ReactNode;
};

export const UiEditorToolbar = forwardRef<HTMLDivElement, UiEditorToolbarProps>(function UiEditorToolbar(
  { className, left, center, right, ...props },
  ref,
) {
  const classes = ['main-toolbar', 'ui-editor-toolbar', className].filter(Boolean).join(' ');
  return (
    <div className={classes} ref={ref} {...props}>
      <div className="toolbar-left">{left}</div>
      <div className="toolbar-center">{center}</div>
      <div className="toolbar-right">{right}</div>
    </div>
  );
});

export function UiToolbarGroup({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  const classes = ['ui-toolbar-group', 'toolbar-group', className].filter(Boolean).join(' ');
  return <div className={classes} {...props} />;
}

export function UiToolbarSeparator(props: HTMLAttributes<HTMLSpanElement>) {
  const { className, ...rest } = props;
  const classes = ['toolbar-separator', className].filter(Boolean).join(' ');
  return <span aria-hidden="true" className={classes} {...rest} />;
}
