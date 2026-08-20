import { X } from 'lucide-react';

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
  return (
    <div className="editor-document-tabs" role="tablist" aria-label="Open documents">
      {documents.map((document) => {
        const registration = registry[document.kind];
        const Icon = registration.icon;
        const active = document.id === activeDocumentId;
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
            {onClose && (
              <button
                aria-label={`Close ${document.title}`}
                className="editor-document-tab-close"
                onClick={() => onClose(document.id)}
                type="button"
              >
                <X size={12} />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
