import { describe, expect, it } from 'vitest';
import { forgetWindowViewportEntries } from './sharedViewportLifecycle';

describe('shared viewport shutdown cleanup', () => {
  it('forgets only the closed window entries using captured webContents identity', () => {
    const targets = new Map([['closed-a', 11], ['closed-b', 11], ['other', 12]]);
    const presented = new Set(['closed-a', 'closed-b', 'other']);
    const failures = new Map([['closed-a', 'error'], ['other', 'retained']]);
    const queued = new Map([['closed-b', Promise.resolve()], ['other', Promise.resolve()]]);
    forgetWindowViewportEntries(11, targets, [presented, failures, queued]);
    expect([...targets]).toEqual([['other', 12]]);
    expect([...presented]).toEqual(['other']);
    expect([...failures]).toEqual([['other', 'retained']]);
    expect([...queued.keys()]).toEqual(['other']);
    expect(() => forgetWindowViewportEntries(11, targets, [presented, failures, queued])).not.toThrow();
  });
});
