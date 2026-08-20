import { useState } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';

import { UiButton } from '../ui';
import { closeEditorDocumentInStore } from './editorDocuments';
import type { EditorDocument, EditorRegistry } from './editorTypes';

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
          const Icon = registration.icon;
          const active = document.id === activeDocumentId;
          const closeable = registration.closeable ?? registration.allowMultiple;
          return (
            <div className={`editor-document-tab${active ? ' active' : ''}`} key={document.id}>
              <button
                aria-selected={active}
                className="editor-document-tab-main"
                onClick={() => onActivate(document.id)}
                role="tab"
                title={[
                  document.path || document.title,
                  registration.title,
                  document.readOnly && 'Read-only',
                  document.recovered && 'Recovered',
                ]
                  .filter(Boolean)
                  .join('\n')}
                type="button"
              >
                <Icon size={13} />
                <span>{document.title}</span>
                {document.readOnly && <small>RO</small>}
                {document.recovered && <small>Recovered</small>}
                {document.dirty && <b aria-label="Unsaved changes">●</b>}
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
