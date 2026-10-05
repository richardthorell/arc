import { describe, expect, it, vi } from 'vitest';

import { EditorCommandExecutor } from './editorCommandExecution';
import {
  deriveEditorCommandPaletteExecutionView,
  executeSelectedEditorCommand,
} from './editorCommandPaletteExecution';
import { EditorCommandRegistry } from './editorCommands';

const createRegistry = (): EditorCommandRegistry => {
  const registry = new EditorCommandRegistry();
  registry.register({ id: 'scene.focus-selection', title: 'Focus Selection', category: 'Scene' });
  registry.register({ id: 'edit.undo', title: 'Undo', category: 'Edit' });
  return registry;
};

describe('editor command palette execution', () => {
  it('projects executable and context-sensitive enabled state for palette matches', () => {
    const registry = createRegistry();
    const executor = new EditorCommandExecutor();
    executor.register('scene.focus-selection', () => undefined, (context) => context.selectionCount === 1);

    const view = deriveEditorCommandPaletteExecutionView(
      registry,
      executor,
      { query: '', selectedCommandId: 'scene.focus-selection' },
      { selectionCount: 0 },
    );

    expect(view.selectedCommandState).toMatchObject({ executable: true, enabled: false });
    expect(view.commandStates.find((state) => state.command.id === 'edit.undo')).toMatchObject({
      executable: false,
      enabled: false,
    });
  });

  it('executes the selected command by stable id through the shared executor', async () => {
    const registry = createRegistry();
    const executor = new EditorCommandExecutor();
    const handler = vi.fn();
    executor.register('scene.focus-selection', handler);

    await expect(
      executeSelectedEditorCommand(registry, executor, {
        query: 'focus',
        selectedCommandId: 'scene.focus-selection',
      }),
    ).resolves.toBe(true);
    expect(handler).toHaveBeenCalledOnce();
  });

  it('does not execute when filtering leaves no selected command', async () => {
    const registry = createRegistry();
    const executor = new EditorCommandExecutor();
    const handler = vi.fn();
    executor.register('scene.focus-selection', handler);

    await expect(executeSelectedEditorCommand(registry, executor, { query: 'missing' })).resolves.toBe(false);
    expect(handler).not.toHaveBeenCalled();
  });
});
