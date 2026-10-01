import { forwardRef } from 'react';
import type { HTMLAttributes, ReactNode } from 'react';

import '../layout/toolbar.css';
import type { EditorToolbarRegion } from './editorToolbarContract';

export type UiEditorToolbarProps = Omit<HTMLAttributes<HTMLDivElement>, 'children'> & {
  left?: ReactNode;
  center?: ReactNode;
  right?: ReactNode;
};

export type UiToolbarRegionProps = HTMLAttributes<HTMLDivElement> & {
  region: EditorToolbarRegion;
};

export function UiToolbarRegion({ className, region, ...props }: UiToolbarRegionProps) {
  const classes = [`toolbar-${region}`, 'ui-toolbar-region', className].filter(Boolean).join(' ');
  return <div className={classes} data-toolbar-region={region} {...props} />;
}

export const UiEditorToolbar = forwardRef<HTMLDivElement, UiEditorToolbarProps>(function UiEditorToolbar(
  { className, left, center, right, role = 'toolbar', ...props },
  ref,
) {
  const classes = ['main-toolbar', 'ui-editor-toolbar', className].filter(Boolean).join(' ');
  return (
    <div className={classes} ref={ref} role={role} {...props}>
      <UiToolbarRegion region="left">{left}</UiToolbarRegion>
      <UiToolbarRegion region="center">{center}</UiToolbarRegion>
      <UiToolbarRegion region="right">{right}</UiToolbarRegion>
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
