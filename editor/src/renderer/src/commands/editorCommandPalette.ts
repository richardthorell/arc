import type { EditorCommand, EditorCommandMatch } from './editorCommands';
import { EditorCommandRegistry } from './editorCommands';

export type EditorCommandPaletteState = {
  query: string;
  selectedCommandId?: string;
};

export type EditorCommandPaletteView = {
  matches: EditorCommandMatch[];
  selectedIndex: number;
  selectedCommand?: EditorCommand;
};

const clampSelection = (index: number, count: number): number => {
  if (count === 0) return -1;
  return Math.min(Math.max(index, 0), count - 1);
};

/**
 * Derive palette presentation from stable command identity rather than list position.
 * This keeps keyboard selection deterministic while filtering or while domains register
 * commands in a different order.
 */
export const deriveEditorCommandPaletteView = (
  registry: EditorCommandRegistry,
  state: EditorCommandPaletteState,
): EditorCommandPaletteView => {
  const matches = registry.search(state.query);
  const selectedIndex = state.selectedCommandId
    ? matches.findIndex((match) => match.command.id === state.selectedCommandId)
    : matches.length > 0
      ? 0
      : -1;
  const resolvedIndex = selectedIndex >= 0 ? selectedIndex : matches.length > 0 ? 0 : -1;
  return {
    matches,
    selectedIndex: resolvedIndex,
    selectedCommand: resolvedIndex >= 0 ? matches[resolvedIndex].command : undefined,
  };
};

export const moveEditorCommandPaletteSelection = (
  registry: EditorCommandRegistry,
  state: EditorCommandPaletteState,
  delta: number,
): EditorCommandPaletteState => {
  const view = deriveEditorCommandPaletteView(registry, state);
  if (view.matches.length === 0) return { ...state, selectedCommandId: undefined };
  const start = view.selectedIndex < 0 ? 0 : view.selectedIndex;
  const selectedIndex = clampSelection(start + delta, view.matches.length);
  return { ...state, selectedCommandId: view.matches[selectedIndex].command.id };
};

export const updateEditorCommandPaletteQuery = (
  registry: EditorCommandRegistry,
  state: EditorCommandPaletteState,
  query: string,
): EditorCommandPaletteState => {
  const next = { ...state, query };
  const view = deriveEditorCommandPaletteView(registry, next);
  return { ...next, selectedCommandId: view.selectedCommand?.id };
};
