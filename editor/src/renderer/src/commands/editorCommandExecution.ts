import type { EditorCommand } from './editorCommands';

export type EditorCommandExecutionContext = Readonly<Record<string, unknown>>;
export type EditorCommandHandler = (context: EditorCommandExecutionContext) => void | Promise<void>;

export type EditorCommandExecutionState = {
  command: EditorCommand;
  executable: boolean;
  enabled: boolean;
};

type RegisteredHandler = {
  handler: EditorCommandHandler;
  isEnabled?: (context: EditorCommandExecutionContext) => boolean;
};

/**
 * Stable execution boundary for editor commands.
 *
 * UI surfaces and automation execute commands by the same persistent command ID instead
 * of importing domain callbacks directly. Registration remains separate from command
 * metadata so discovery/palette code stays presentation-only.
 */
export class EditorCommandExecutor {
  private readonly handlers = new Map<string, RegisteredHandler>();

  register(
    commandId: string,
    handler: EditorCommandHandler,
    isEnabled?: (context: EditorCommandExecutionContext) => boolean,
  ): void {
    const id = commandId.trim();
    if (!id) throw new Error('Editor command handler ID must not be empty');
    if (this.handlers.has(id)) throw new Error(`Editor command handler already registered: ${id}`);
    this.handlers.set(id, { handler, isEnabled });
  }

  unregister(commandId: string): void {
    this.handlers.delete(commandId);
  }

  state(command: EditorCommand, context: EditorCommandExecutionContext = {}): EditorCommandExecutionState {
    const registration = this.handlers.get(command.id);
    return {
      command,
      executable: registration !== undefined,
      enabled: registration !== undefined && (registration.isEnabled?.(context) ?? true),
    };
  }

  async execute(commandId: string, context: EditorCommandExecutionContext = {}): Promise<boolean> {
    const registration = this.handlers.get(commandId);
    if (!registration || !(registration.isEnabled?.(context) ?? true)) return false;
    await registration.handler(context);
    return true;
  }
}
