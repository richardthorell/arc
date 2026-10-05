import { describe, expect, it, vi } from 'vitest';

import { EditorCommandExecutor } from './editorCommandExecution';
import type { EditorCommand } from './editorCommands';

const command: EditorCommand = {
  id: 'scene.focus-selection',
  title: 'Focus Selection',
  category: 'Scene',
};

describe('EditorCommandExecutor', () => {
  it('executes registered handlers by stable command id', async () => {
    const executor = new EditorCommandExecutor();
    const handler = vi.fn();
    const context = { selectionCount: 2 };
    executor.register(command.id, handler);

    await expect(executor.execute(command.id, context)).resolves.toBe(true);
    expect(handler).toHaveBeenCalledOnce();
    expect(handler).toHaveBeenCalledWith(context);
  });

  it('reports and respects enabled state without invoking disabled commands', async () => {
    const executor = new EditorCommandExecutor();
    const handler = vi.fn();
    executor.register(command.id, handler, (context) => context.selectionCount === 1);

    expect(executor.state(command, { selectionCount: 0 })).toEqual({
      command,
      executable: true,
      enabled: false,
    });
    await expect(executor.execute(command.id, { selectionCount: 0 })).resolves.toBe(false);
    expect(handler).not.toHaveBeenCalled();
  });

  it('treats commands without handlers as discoverable but not executable', async () => {
    const executor = new EditorCommandExecutor();

    expect(executor.state(command)).toEqual({ command, executable: false, enabled: false });
    await expect(executor.execute(command.id)).resolves.toBe(false);
  });

  it('rejects duplicate handler registration and supports explicit unregister', async () => {
    const executor = new EditorCommandExecutor();
    executor.register(command.id, () => undefined);

    expect(() => executor.register(command.id, () => undefined)).toThrow(
      'Editor command handler already registered: scene.focus-selection',
    );

    executor.unregister(command.id);
    await expect(executor.execute(command.id)).resolves.toBe(false);
  });
});
