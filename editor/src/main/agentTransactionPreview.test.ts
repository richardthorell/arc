import { describe, expect, it } from 'vitest';
import { buildAgentTransactionPreview, type AgentTransactionChange } from './agentTransactionPreview';

const policy = { mediumChangeCount: 2, highChangeCount: 4 } as const;

describe('buildAgentTransactionPreview', () => {
  it('describes the exact revision-owned change without mutating source state', () => {
    const before = { name: 'Cube', transform: { x: 0 } };
    const after = { name: 'Cube', transform: { x: 4 } };
    const changes: AgentTransactionChange[] = [
      { kind: 'update', domain: 'scene', targetId: 'entity-1', before, after },
    ];

    const preview = buildAgentTransactionPreview('tx-1', 'scene-r7', changes, policy);
    (before.transform as { x: number }).x = 99;
    changes[0] = { ...changes[0], targetId: 'changed-after-preview' };

    expect(preview.transactionId).toBe('tx-1');
    expect(preview.expectedRevision).toBe('scene-r7');
    expect(preview.changes[0].targetId).toBe('entity-1');
    expect(preview.changes[0].before).toEqual({ name: 'Cube', transform: { x: 0 } });
    expect(preview.risk).toBe('low');
    expect(preview.requiresApproval).toBe(false);
  });

  it('makes broad edits recognizable before approval', () => {
    const changes: AgentTransactionChange[] = [
      { kind: 'create', domain: 'component', targetId: 'c1', after: { type: 'Light' } },
      { kind: 'update', domain: 'asset', targetId: 'a1', before: { value: 1 }, after: { value: 2 } },
    ];

    const preview = buildAgentTransactionPreview('tx-2', 'project-r9', changes, policy);
    expect(preview.affectedTargets).toBe(2);
    expect(preview.risk).toBe('medium');
    expect(preview.requiresApproval).toBe(true);
    expect(preview.summary).toBe('2 targets: 1 create, 1 update');
  });

  it('treats destructive edits as high risk even when narrow', () => {
    const preview = buildAgentTransactionPreview(
      'tx-delete',
      'asset-r3',
      [{ kind: 'delete', domain: 'asset', targetId: 'asset-7', before: { path: 'old.mat' } }],
      policy,
    );
    expect(preview.risk).toBe('high');
    expect(preview.requiresApproval).toBe(true);
  });

  it('rejects malformed or ambiguous transaction changes', () => {
    expect(() => buildAgentTransactionPreview('tx', 'r1', [], policy)).toThrow('at least one change');
    expect(() =>
      buildAgentTransactionPreview(
        'tx',
        'r1',
        [{ kind: 'update', domain: 'graph', targetId: 'node-1', before: { x: 1 } }],
        policy,
      ),
    ).toThrow('before and after');
    expect(() =>
      buildAgentTransactionPreview(
        'tx',
        'r1',
        [
          { kind: 'create', domain: 'scene', targetId: 'same', after: {} },
          { kind: 'update', domain: 'scene', targetId: 'same', before: {}, after: {} },
        ],
        policy,
      ),
    ).toThrow('duplicate transaction target');
  });
});
