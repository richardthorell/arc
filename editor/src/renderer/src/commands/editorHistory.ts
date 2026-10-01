export type EditorHistoryEntry = {
  id: string;
  commandId: string;
  title: string;
  timestamp: number;
  reversible: boolean;
};

export type EditorHistoryPresentationEntry = EditorHistoryEntry & {
  position: number;
  isCurrent: boolean;
  isApplied: boolean;
};

export type EditorHistoryPresentation = {
  entries: EditorHistoryPresentationEntry[];
  canUndo: boolean;
  canRedo: boolean;
  undoTitle?: string;
  redoTitle?: string;
};

/**
 * Build a deterministic, UI-ready history snapshot without exposing mutable undo-stack state.
 *
 * `cursor` is the number of currently applied entries. A cursor at entries.length means
 * everything is applied; moving it backwards exposes redo candidates. Entries are kept in
 * their authored order so history UI communicates the exact transaction sequence.
 */
export const presentEditorHistory = (
  entries: readonly EditorHistoryEntry[],
  cursor: number,
): EditorHistoryPresentation => {
  const safeCursor = Math.max(0, Math.min(Math.trunc(cursor), entries.length));
  const presented = entries.map((entry, index): EditorHistoryPresentationEntry => ({
    ...entry,
    position: index,
    isCurrent: safeCursor > 0 && index === safeCursor - 1,
    isApplied: index < safeCursor,
  }));

  let undoTitle: string | undefined;
  for (let index = safeCursor - 1; index >= 0; index -= 1) {
    if (entries[index].reversible) {
      undoTitle = entries[index].title;
      break;
    }
  }

  let redoTitle: string | undefined;
  for (let index = safeCursor; index < entries.length; index += 1) {
    if (entries[index].reversible) {
      redoTitle = entries[index].title;
      break;
    }
  }

  return {
    entries: presented,
    canUndo: undoTitle !== undefined,
    canRedo: redoTitle !== undefined,
    undoTitle,
    redoTitle,
  };
};
