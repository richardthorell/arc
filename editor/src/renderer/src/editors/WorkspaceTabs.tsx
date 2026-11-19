import { useState } from 'react';
import { createPortal } from 'react-dom';
import { Activity, PanelsTopLeft, X } from 'lucide-react';

import { DocumentTypeIcon, type DocumentTypeIconKind } from '../assets/DocumentTypeIcon';
import { UiButton } from '../ui';
import { closeEditorDocumentInStore } from '../editors/editorDocuments';
import type { EditorDocument, EditorDocumentKind, EditorRegistry } from '../editors/editorTypes';
import type { WorkspaceTab } from '../app/workspaceTabs';

const documentIconKinds: Record<EditorDocumentKind, DocumentTypeIconKind> = {
  level: 'level',
  shader: 'shader',
  material: 'material',
  materialInstance: 'material',
  materialFunction: 'material',
  sound: 'audio',
  flow: 'script',
  texture: 'texture',
  model: 'model',
  skeleton: 'skeleton',
};

/** Shared tab chrome. Assets retain dirty/save semantics; tool views have independent lifecycles. */
export function WorkspaceTabs({
  tabs,
  activeTabId,
  registry,
  onActivate,
  onClose,
}: {
  tabs: readonly WorkspaceTab[];
  activeTabId: string | null;
  registry: EditorRegistry;
  onActivate: (tabId: string) => void;
  onClose?: (tabId: string) => void;
}) {
  const [pendingCloseId, setPendingCloseId] = useState<string | null>(null);
  const pendingClose = tabs.find((tab) => tab.type === 'asset' && tab.id === pendingCloseId);
  const pendingDocument = pendingClose?.type === 'asset' ? pendingClose.document : null;

  const closeTab = (tab: WorkspaceTab) => {
    if (tab.type === 'asset') {
      registry[tab.document.kind].onClosed?.(tab.document);
      if (!onClose) closeEditorDocumentInStore(tab.id);
      else onClose(tab.id);
    } else {
      onClose?.(tab.id);
    }
    setPendingCloseId(null);
  };

  const requestClose = (tab: WorkspaceTab) => {
    if (tab.type === 'asset' && tab.document.dirty) {
      setPendingCloseId(tab.id);
      return;
    }
    closeTab(tab);
  };

  return (
    <>
      <div className="editor-document-tabs" role="tablist" aria-label="Workspace tabs">
        {tabs.map((tab) => {
          const document = tab.type === 'asset' ? tab.document : null;
          const registration = document ? registry[document.kind] : null;
          const title = document?.title ?? (tab.type === 'view' ? tab.title : '');
          const active = tab.id === activeTabId;
          const closeable = tab.type === 'view' ? tab.closeable : (registration?.closeable ?? registration?.allowMultiple ?? true);
          const ViewIcon = tab.type === 'view' ? (tab.viewId === 'builds' ? Activity : PanelsTopLeft) : null;
          return (
            <div
              className={`editor-document-tab${active ? ' active' : ''}${document?.dirty ? ' dirty' : ''}`}
              key={tab.id}
              onMouseDown={(event) => {
                if (event.button !== 1 || !closeable) return;
                event.preventDefault();
                requestClose(tab);
              }}
            >
              <button
                aria-selected={active}
                className="editor-document-tab-main"
                onClick={() => onActivate(tab.id)}
                role="tab"
                title={[
                  document?.path || title,
                  registration?.title ?? (tab.type === 'view' ? 'Workspace view' : ''),
                  document?.dirty && 'Unsaved changes',
                  document?.readOnly && 'Read-only',
                  document?.recovered && 'Recovered',
                ]
                  .filter(Boolean)
                  .join('\n')}
                type="button"
              >
                {document && (
                  <DocumentTypeIcon
                    className="editor-document-tab-icon"
                    kind={documentIconKinds[document.kind]}
                    size={15}
                  />
                )}
                {ViewIcon && <ViewIcon aria-hidden="true" className="editor-document-tab-icon" size={15} />}
                <span className="editor-document-tab-title">{title}</span>
                {document?.readOnly && <small>RO</small>}
                {document?.recovered && <small>Recovered</small>}
                {document?.dirty && (
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
                  aria-label={`Close ${title}`}
                  className="editor-document-tab-close"
                  onClick={() => requestClose(tab)}
                  type="button"
                >
                  <X size={12} />
                </button>
              )}
            </div>
          );
        })}
      </div>
      {pendingDocument &&
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
                <strong>{pendingDocument.title}</strong> has unsaved changes.
              </p>
              <div className="editor-document-close-actions">
                <UiButton
                  disabled={!registry[pendingDocument.kind].save}
                  onClick={() => {
                    const registration = registry[pendingDocument.kind];
                    if (!registration.save) return;
                    void registration.save(pendingDocument).then((saved) => {
                      if (saved)
                        closeTab({
                          type: 'asset',
                          id: pendingDocument.id,
                          document: { ...pendingDocument, dirty: false },
                        });
                    });
                  }}
                  variant="primary"
                >
                  Save
                </UiButton>
                <UiButton
                  onClick={() => closeTab({ type: 'asset', id: pendingDocument.id, document: pendingDocument })}
                  variant="default"
                >
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

/** Old asset-only callers continue using the shared workspace tab renderer. */
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
    <WorkspaceTabs
      tabs={documents.map((document) => ({ type: 'asset', id: document.id, document }))}
      activeTabId={activeDocumentId}
      registry={registry}
      onActivate={onActivate}
      onClose={onClose}
    />
  );
}
