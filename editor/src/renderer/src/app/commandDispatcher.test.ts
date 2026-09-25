import { afterEach, describe, expect, it, vi } from 'vitest';

import { dispatchWorkbenchCommand, registerWorkbenchCommandHandler } from './commandDispatcher';

let cleanup: (() => void) | undefined;

afterEach(() => {
  cleanup?.();
  cleanup = undefined;
});

describe('commandDispatcher', () => {
  it('dispatches to the registered workbench handler', () => {
    const handler = vi.fn();
    cleanup = registerWorkbenchCommandHandler(handler);

    expect(dispatchWorkbenchCommand('file.save')).toBe(true);
    expect(handler).toHaveBeenCalledWith('file.save');
  });

  it('stops dispatching when the active handler unregisters', () => {
    cleanup = registerWorkbenchCommandHandler(() => undefined);
    cleanup();
    cleanup = undefined;

    expect(dispatchWorkbenchCommand('file.save')).toBe(false);
  });
});
