import { forwardRef } from 'react';
import type { HTMLAttributes, ReactNode } from 'react';

import '../layout/toolbar.css';

export type UiEditorToolbarProps = Omit<HTMLAttributes<HTMLDivElement>, 'children'> & {
  left?: ReactNode;
  center?: ReactNode;
  right?: ReactNode;
};

export const UiEditorToolbar = forwardRef<HTMLDivElement, UiEditorToolbarProps>(function UiEditorToolbar(
  { className, left, center, right, role = 'toolbar', ...props },
  ref,
) {
  const classes = ['main-toolbar', 'ui-editor-toolbar', className].filter(Boolean).join(' ');
  return (
    <div className={classes} ref={ref} role={role} {...props}>
      <div className="toolbar-left" data-toolbar-region="left">
        {left}
      </div>
      <div className="toolbar-center" data-toolbar-region="center">
        {center}
      </div>
      <div className="toolbar-right" data-toolbar-region="right">
        {right}
      </div>
    </div>
  );
});

export function UiToolbarGroup({ className, role = 'group', ...props }: HTMLAttributes<HTMLDivElement>) {
  const classes = ['ui-toolbar-group', 'toolbar-group', className].filter(Boolean).join(' ');
  return <div className={classes} role={role} {...props} />;
}

export function UiToolbarSeparator(props: HTMLAttributes<HTMLSpanElement>) {
  const { className, ...rest } = props;
  const classes = ['toolbar-separator', className].filter(Boolean).join(' ');
  return <span aria-hidden="true" className={classes} {...rest} />;
}
