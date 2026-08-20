import type { ReactNode } from 'react';

import type { WorkbenchIcon } from '../app/workbenchTypes';

export type EditorDocumentKind = 'level';

export type EditorDocument = {
  id: string;
  kind: EditorDocumentKind;
  title: string;
  path?: string;
  dirty: boolean;
  readOnly: boolean;
  recovered?: boolean;
};

export type EditorSurfaceContext = {
  instanceId?: string;
  onMaximizeToggle?: () => void;
};

export type EditorRegistration = {
  kind: EditorDocumentKind;
  title: string;
  icon: WorkbenchIcon;
  allowMultiple: boolean;
  render: (document: EditorDocument, context: EditorSurfaceContext) => ReactNode;
  renderToolbar: (document: EditorDocument) => ReactNode;
};

export type EditorRegistry = Readonly<Record<EditorDocumentKind, EditorRegistration>>;
