import { useState } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';

import { DocumentTypeIcon, type DocumentTypeIconKind } from '../assets/DocumentTypeIcon';
import { UiButton } from '../ui';
import { closeEditorDocumentInStore } from './editorDocuments';
import type { EditorDocument, EditorDocumentKind, EditorRegistry } from './editorTypes';

const documentIconKinds: Record<EditorDocumentKind, DocumentTypeIconKind> = {
  level: 'level',
  shader: 'shader',
  material: 'material',
  flow: 'script',
  texture: 'texture',
  model: 'model',
  skeleton: 'skeleton',
};

export function EditorDocumentTabs({
  documents,
  activeDocumentId,
  registry,
  onActivate,
  onClose,
}: {
  documents: readonly EditorDocument[];
  activeDocumentId: string | null;
  registry: EditorRegistry;
  onActivate: (documentId: string) => void;
  onClose?: (documentId: string) => void;
}) {
  const [pendingCloseId, setPendingCloseId] = useState<string | null>(null);
  const pendingClose = documents.find((document) => document.id === pendingCloseId) ?? null;

  const closeDocument = (document: EditorDocument) => {
    const registration = registry[document.kind];
    registration.onClosed?.(document);
    if (onClose) onClose(document.id);
    else closeEditorDocumentInStore(document.id);
    setPendingCloseId(null);
  };

  const requestClose = (document: EditorDocument) => {
    if (document.dirty) {
      setPendingCloseId(document.id);
      return;
    }
    closeDocument(document);
  };

  return (
    <>
      <div className="editor-document-tabs" role="tablist" aria-label="Open documents">
        {documents.map((document) => {
          const registration = registry[document.kind];
          const active = document.id === activeDocumentId;
          const closeable = registration.closeable ?? registration.allowMultiple;
          return (
            <div
              className={`editor-document-tab${active ? ' active' : ''}${document.dirty ? ' dirty' : ''}`}
              key={document.id}
              onMouseDown={(event) => {
                if (event.button !== 1 || !closeable) return;
                event.preventDefault();
                requestClose(document);
              }}
            >
              <button
                aria-selected={active}
                className="editor-document-tab-main"
                onClick={() => onActivate(document.id)}
                role="tab"
                title={[
                  document.path || document.title,
                  registration.title,
                  document.dirty && 'Unsaved changes',
                  document.readOnly && 'Read-only',
                  document.recovered && 'Recovered',
                ]
                  .filter(Boolean)
                  .join('\n')}
                type="button"
              >
                <DocumentTypeIcon
                  className="editor-document-tab-icon"
                  kind={documentIconKinds[document.kind]}
                  size={15}
                />
                <span className="editor-document-tab-title">{document.title}</span>
                {document.readOnly && <small>RO</small>}
                {document.recovered && <small>Recovered</small>}
                {document.dirty && (
                  <span
                    aria-label="Unsaved changes"
                    className="editor-document-tab-dirty-indicator"
                    title="Unsaved changes"
                  >
                    ●
                  </span>
                )}
              </button>
              {closeable && (
                <button
                  aria-label={`Close ${document.title}`}
                  className="editor-document-tab-close"
                  onClick={() => requestClose(document)}
                  type="button"
                >
                  <X size={12} />
                </button>
              )}
            </div>
          );
        })}
      </div>
      {pendingClose &&
        createPortal(
          <div className="editor-document-close-backdrop" role="presentation">
            <section
              aria-labelledby="editor-document-close-title"
              aria-modal="true"
              className="editor-document-close-dialog"
              role="dialog"
            >
              <h2 id="editor-document-close-title">Save changes?</h2>
              <p>
                <strong>{pendingClose.title}</strong> has unsaved changes.
              </p>
              <div className="editor-document-close-actions">
                <UiButton
                  disabled={!registry[pendingClose.kind].save}
                  onClick={() => {
                    const registration = registry[pendingClose.kind];
                    if (!registration.save) return;
                    void registration.save(pendingClose).then((saved) => {
                      if (saved) closeDocument({ ...pendingClose, dirty: false });
                    });
                  }}
                  variant="primary"
                >
                  Save
                </UiButton>
                <UiButton onClick={() => closeDocument(pendingClose)} variant="default">
                  Don't Save
                </UiButton>
                <UiButton onClick={() => setPendingCloseId(null)} variant="ghost">
                  Cancel
                </UiButton>
              </div>
            </section>
          </div>,
          document.body,
        )}
    </>
  );
}
