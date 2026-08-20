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
  if (!document) return <div className="editor-toolbar-host editor-toolbar-host-empty" />;
  const registration = getEditorRegistration(registry, document.kind);
  return (
    <div className="editor-toolbar-host" data-editor-kind={document.kind}>
      {registration.renderToolbar(document)}
    </div>
  );
}
