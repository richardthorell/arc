import { useEffect } from 'react';

import type { EditorDocument, EditorRegistry, EditorSurfaceContext } from './editorTypes';
import { getEditorRegistration } from './editorRegistry';

export function EditorHost({
  document,
  registry,
  context = {},
}: {
  document: EditorDocument | null;
  registry: EditorRegistry;
  context?: EditorSurfaceContext;
}) {
  if (!document) return <div className="editor-host editor-host-empty">No document open</div>;
  const registration = getEditorRegistration(registry, document.kind);

  if (document.kind !== 'level' && context.instanceId) {
    return (
      <div className="editor-host editor-host-empty" data-editor-kind={document.kind}>
        Auxiliary viewports are available in the Level Editor.
      </div>
    );
  }

  return (
    <div className="editor-host" data-editor-kind={document.kind}>
      {registration.render(document, context)}
    </div>
  );
}

export function EditorToolbarHost({
  document,
  registry,
}: {
  document: EditorDocument | null;
  registry: EditorRegistry;
}) {
  const registration = document ? getEditorRegistration(registry, document.kind) : null;

  useEffect(() => {
    if (!document || !registration?.save || document.kind === 'level') return;
    const save = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.shiftKey || event.altKey || event.key.toLocaleLowerCase() !== 's')
        return;
      event.preventDefault();
      event.stopImmediatePropagation();
      void registration.save?.(document);
    };
    window.addEventListener('keydown', save, true);
    return () => window.removeEventListener('keydown', save, true);
  }, [document, registration]);

  if (!document || !registration) return <div className="editor-toolbar-host editor-toolbar-host-empty" />;
  return (
    <div className="editor-toolbar-host" data-editor-kind={document.kind}>
      {registration.renderToolbar(document)}
    </div>
  );
}
