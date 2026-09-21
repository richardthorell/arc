import { createContext, useContext, type ReactNode } from 'react';

const EditorSurfaceActivity = createContext(true);

export function EditorSurfaceActivityProvider({ active, children }: { active: boolean; children: ReactNode }) {
  return <EditorSurfaceActivity.Provider value={active}>{children}</EditorSurfaceActivity.Provider>;
}

export const useEditorSurfaceActive = () => useContext(EditorSurfaceActivity);
