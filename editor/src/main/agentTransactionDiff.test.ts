import { describe, expect, it } from 'vitest';

import { createAgentTransactionDiffPreview } from './agentTransactionDiff';

describe('createAgentTransactionDiffPreview', () => {
  it('summarizes small scene edits as low risk', () => {
    const preview = createAgentTransactionDiffPreview([
      {
        kind: 'scene',
        operation: 'setTransform',
        target: { guid: 'entity-1', component: 'transform' },
        before: { position: [0, 0, 0] },
        after: { position: [1, 0, 0] },
      },
    ]);

    expect(preview).toMatchObject({
      changeCount: 1,
      sceneChangeCount: 1,
      assetChangeCount: 0,
      destructiveChangeCount: 0,
      risk: 'low',
      summary: '1 scene change',
    });
  });

  it('keeps stable asset identity and marks asset edits as medium risk', () => {
    const preview = createAgentTransactionDiffPreview([
      {
        kind: 'asset',
        operation: 'createAsset',
        target: { path: 'Materials/Robot.arcmat' },
        after: { revision: 'abc123', kind: 'material' },
      },
    ]);

    expect(preview.risk).toBe('medium');
    expect(preview.summary).toBe('1 asset change');
    expect(preview.changes[0].target.path).toBe('Materials/Robot.arcmat');
  });

  it('marks destructive edits as high risk', () => {
    const preview = createAgentTransactionDiffPreview([
      {
        kind: 'scene',
        operation: 'delete',
        target: { guid: 'entity-2' },
        before: { name: 'Important Entity' },
      },
    ]);

    expect(preview.destructiveChangeCount).toBe(1);
    expect(preview.risk).toBe('high');
  });

  it('marks broad edits as high risk even when individually non-destructive', () => {
    const preview = createAgentTransactionDiffPreview(
      Array.from({ length: 20 }, (_, index) => ({
        kind: 'scene' as const,
        operation: 'rename',
        target: { guid: `entity-${index}` },
        before: { name: `Before ${index}` },
        after: { name: `After ${index}` },
      })),
    );

    expect(preview.changeCount).toBe(20);
    expect(preview.risk).toBe('high');
  });

  it('returns an explicit empty preview', () => {
    expect(createAgentTransactionDiffPreview([])).toEqual({
      changeCount: 0,
      sceneChangeCount: 0,
      assetChangeCount: 0,
      destructiveChangeCount: 0,
      risk: 'low',
      summary: 'No changes',
      changes: [],
    });
  });
});
