import type { ReactNode } from 'react';

import type { AssetItem } from '../services/editorHostTypes';
import type { WorkbenchIcon } from '../app/workbenchTypes';

export type EditorDocumentKind = 'level' | 'shader';

export type EditorDocument = {
  id: string;
  kind: EditorDocumentKind;
  title: string;
  path?: string;
  assetId?: string;
  assetGuid?: string;
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
  closeable?: boolean;
  canOpenAsset?: (asset: AssetItem) => boolean;
  createDocument?: (asset: AssetItem) => EditorDocument;
  render: (document: EditorDocument, context: EditorSurfaceContext) => ReactNode;
  renderToolbar: (document: EditorDocument) => ReactNode;
  save?: (document: EditorDocument) => Promise<boolean>;
  onClosed?: (document: EditorDocument) => void;
};

export type EditorRegistry = Readonly<Record<EditorDocumentKind, EditorRegistration>>;
export type EditorRegistrySeed = Pick<EditorRegistry, 'level'> & Partial<EditorRegistry>;
