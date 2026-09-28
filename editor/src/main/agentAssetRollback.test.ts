import { describe, expect, it } from 'vitest';

import { planAgentAssetRollback, validateAgentAssetRollbackRecord } from './agentAssetRollback';

describe('agent asset rollback', () => {
  it('removes an unchanged asset created by an agent operation', () => {
    expect(
      planAgentAssetRollback({
        record: {
          operationId: 'op-1',
          path: 'Materials/New.arcmat',
          operation: 'create',
          committedRevision: 'created-revision',
        },
        currentRevision: 'created-revision',
      }),
    ).toEqual({
      ok: true,
      action: 'remove',
      path: 'Materials/New.arcmat',
      expectedRevision: 'created-revision',
    });
  });

  it('restores prior content only when a replaced asset is still at the committed revision', () => {
    expect(
      planAgentAssetRollback({
        record: {
          operationId: 'op-2',
          path: 'Flow/Player.arcflow',
          operation: 'replace',
          beforeRevision: 'before',
          beforeContent: '{"version":1}',
          committedRevision: 'after',
        },
        currentRevision: 'after',
      }),
    ).toEqual({
      ok: true,
      action: 'restore',
      path: 'Flow/Player.arcflow',
      expectedRevision: 'after',
      content: '{"version":1}',
    });
  });

  it('refuses to overwrite changes made after the committed operation', () => {
    expect(
      planAgentAssetRollback({
        record: {
          operationId: 'op-3',
          path: 'Materials/Changed.arcmat',
          operation: 'replace',
          beforeRevision: 'before',
          beforeContent: 'old',
          committedRevision: 'agent-write',
        },
        currentRevision: 'human-write',
      }),
    ).toEqual({
      ok: false,
      reason: 'Asset revision changed after the agent operation; refusing rollback.',
    });
  });

  it('restores a removed asset only when the path is still vacant', () => {
    const record = {
      operationId: 'op-4',
      path: 'Materials/Removed.arcmat',
      operation: 'remove' as const,
      beforeRevision: 'before',
      beforeContent: 'original',
    };

    expect(planAgentAssetRollback({ record })).toEqual({
      ok: true,
      action: 'restore',
      path: 'Materials/Removed.arcmat',
      content: 'original',
    });
    expect(planAgentAssetRollback({ record, currentRevision: 'replacement' })).toEqual({
      ok: false,
      reason: 'Removed asset path is occupied; refusing rollback.',
    });
  });

  it('rejects incomplete rollback records', () => {
    expect(
      validateAgentAssetRollbackRecord({
        operationId: 'op-5',
        path: 'Materials/Broken.arcmat',
        operation: 'replace',
        committedRevision: 'after',
      }),
    ).toBe('Replaced or removed assets require the prior revision and content.');
  });
});
