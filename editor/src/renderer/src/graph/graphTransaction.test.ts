import { describe, expect, it, vi } from 'vitest';

import { runGraphTransaction, type GraphTransactionHost } from './graphTransaction';

describe('runGraphTransaction', () => {
  it('commits a successful edit exactly once and returns its result', () => {
    const host: GraphTransactionHost<number> = {
      begin: vi.fn(() => 17),
      commit: vi.fn(),
      rollback: vi.fn(),
    };

    const result = runGraphTransaction(host, 'Move nodes', (token) => {
      expect(token).toBe(17);
      return 'moved';
    });

    expect(result).toBe('moved');
    expect(host.begin).toHaveBeenCalledOnce();
    expect(host.begin).toHaveBeenCalledWith('Move nodes');
    expect(host.commit).toHaveBeenCalledOnce();
    expect(host.commit).toHaveBeenCalledWith(17);
    expect(host.rollback).not.toHaveBeenCalled();
  });

  it('rolls back a failed edit and preserves the original failure', () => {
    const failure = new Error('connection rejected');
    const host: GraphTransactionHost<string> = {
      begin: vi.fn(() => 'transaction-1'),
      commit: vi.fn(),
      rollback: vi.fn(),
    };

    expect(() =>
      runGraphTransaction(host, 'Connect pins', () => {
        throw failure;
      }),
    ).toThrow(failure);

    expect(host.commit).not.toHaveBeenCalled();
    expect(host.rollback).toHaveBeenCalledOnce();
    expect(host.rollback).toHaveBeenCalledWith('transaction-1');
  });

  it('does not open a second transaction around one compound edit', () => {
    const events: string[] = [];
    const host: GraphTransactionHost<symbol> = {
      begin: (label) => {
        events.push(`begin:${label}`);
        return Symbol(label);
      },
      commit: () => events.push('commit'),
      rollback: () => events.push('rollback'),
    };

    runGraphTransaction(host, 'Paste nodes', () => {
      events.push('create nodes');
      events.push('create connections');
    });

    expect(events).toEqual(['begin:Paste nodes', 'create nodes', 'create connections', 'commit']);
  });
});
