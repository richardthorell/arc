import { describe, expect, it } from 'vitest';
import { presentEditorHistory, type EditorHistoryEntry } from './editorHistory';

const history: EditorHistoryEntry[] = [
  { id: '1', commandId: 'scene.create', title: 'Create Entity', timestamp: 10, reversible: true },
  { id: '2', commandId: 'scene.select', title: 'Select Entity', timestamp: 20, reversible: false },
  { id: '3', commandId: 'scene.move', title: 'Move Entity', timestamp: 30, reversible: true },
];

describe('presentEditorHistory', () => {
  it('marks the applied range and current entry deterministically', () => {
    const result = presentEditorHistory(history, 2);

    expect(result.entries.map(({ position, isApplied, isCurrent }) => ({ position, isApplied, isCurrent }))).toEqual([
      { position: 0, isApplied: true, isCurrent: false },
      { position: 1, isApplied: true, isCurrent: true },
      { position: 2, isApplied: false, isCurrent: false },
    ]);
  });

  it('describes the nearest reversible undo and redo operations', () => {
    const result = presentEditorHistory(history, 2);

    expect(result.canUndo).toBe(true);
    expect(result.undoTitle).toBe('Create Entity');
    expect(result.canRedo).toBe(true);
    expect(result.redoTitle).toBe('Move Entity');
  });

  it('clamps stale persisted cursors to the available history', () => {
    expect(presentEditorHistory(history, -4).canUndo).toBe(false);
    const afterEnd = presentEditorHistory(history, 99);
    expect(afterEnd.canRedo).toBe(false);
    expect(afterEnd.undoTitle).toBe('Move Entity');
  });

  it('keeps empty history safe for presentation', () => {
    expect(presentEditorHistory([], 0)).toEqual({
      entries: [],
      canUndo: false,
      canRedo: false,
      undoTitle: undefined,
      redoTitle: undefined,
    });
  });
});
