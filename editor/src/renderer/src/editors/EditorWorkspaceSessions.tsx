import type { ReactNode } from 'react';

import type { EditorDocument } from './editorTypes';

export function EditorWorkspaceSessions({
  documents,
  activeDocumentId,
  projectKey,
  renderDocument,
}: {
  documents: EditorDocument[];
  activeDocumentId: string | null;
  projectKey: string;
  renderDocument: (document: EditorDocument, active: boolean) => ReactNode;
}) {
  return documents.map((document) => {
    const active = document.id === activeDocumentId;
    return (
      <div
        aria-hidden={!active}
        className={`editor-workspace-session${active ? ' is-active' : ''}`}
        data-document-id={document.id}
        key={`${projectKey}:${document.id}`}
      >
        {renderDocument(document, active)}
      </div>
    );
  });
}
