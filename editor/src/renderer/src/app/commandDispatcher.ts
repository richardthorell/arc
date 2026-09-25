import type { CommandId } from './workbenchTypes';

type CommandHandler = (command: CommandId) => void;

let activeHandler: CommandHandler | null = null;

export const registerWorkbenchCommandHandler = (handler: CommandHandler) => {
  activeHandler = handler;
  return () => {
    if (activeHandler === handler) activeHandler = null;
  };
};

export const dispatchWorkbenchCommand = (command: CommandId): boolean => {
  if (!activeHandler) return false;
  activeHandler(command);
  return true;
};
