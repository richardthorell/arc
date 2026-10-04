import { createContext, useContext, type ReactNode } from 'react';

import type { EditorReferenceController } from './editorReferences';

const EditorReferenceContext = createContext<EditorReferenceController | null>(null);

export function EditorReferenceProvider({
  controller,
  children,
}: {
  controller: EditorReferenceController;
  children: ReactNode;
}) {
  return <EditorReferenceContext.Provider value={controller}>{children}</EditorReferenceContext.Provider>;
}

export const useEditorReferenceController = () => useContext(EditorReferenceContext);
