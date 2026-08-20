import type { ReactNode } from 'react';

export function LevelEditor({ children }: { children: ReactNode }) {
  return <div className="level-editor">{children}</div>;
}
