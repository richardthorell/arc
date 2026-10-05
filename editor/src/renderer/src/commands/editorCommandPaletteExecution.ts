import {
  EditorCommandExecutor,
  type EditorCommandExecutionContext,
  type EditorCommandExecutionState,
} from './editorCommandExecution';
import { deriveEditorCommandPaletteView, type EditorCommandPaletteState } from './editorCommandPalette';
import { EditorCommandRegistry } from './editorCommands';

export type EditorCommandPaletteExecutionView = {
  commandStates: EditorCommandExecutionState[];
  selectedCommandState?: EditorCommandExecutionState;
};

/**
 * Project command-palette matches through the authoritative command executor so
 * presentation surfaces can distinguish discoverable commands from commands that
 * are currently executable/enabled without duplicating domain enablement logic.
 */
export const deriveEditorCommandPaletteExecutionView = (
  registry: EditorCommandRegistry,
  executor: EditorCommandExecutor,
  state: EditorCommandPaletteState,
  context: EditorCommandExecutionContext = {},
): EditorCommandPaletteExecutionView => {
  const palette = deriveEditorCommandPaletteView(registry, state);
  const commandStates = palette.matches.map((match) => executor.state(match.command, context));
  return {
    commandStates,
    selectedCommandState: palette.selectedCommand ? executor.state(palette.selectedCommand, context) : undefined,
  };
};

/** Execute the selected palette command through the same stable-ID boundary used by automation. */
export const executeSelectedEditorCommand = async (
  registry: EditorCommandRegistry,
  executor: EditorCommandExecutor,
  state: EditorCommandPaletteState,
  context: EditorCommandExecutionContext = {},
): Promise<boolean> => {
  const selected = deriveEditorCommandPaletteView(registry, state).selectedCommand;
  if (!selected) return false;
  return executor.execute(selected.id, context);
};
