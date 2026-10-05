import { describe, expect, it } from 'vitest';

import { parseAgentEditorBatchRequest } from './agentEditorBatch';

describe('agent editor batch contract', () => {
  it('accepts entity creation followed by tempId-based edits', () => {
    expect(
      parseAgentEditorBatchRequest({
        editSessionId: 'edit-1',
        expectedSceneRevision: 4,
        operations: [
          { type: 'entity.create', tempId: 'cube', kind: 'cube' },
          {
            type: 'entity.setTransform',
            target: { tempId: 'cube' },
            transform: {
              position: [0, 0, 0],
              rotation: [0, 0, 0, 1],
              scale: [4, 4, 4],
            },
          },
          { type: 'entity.rename', target: { tempId: 'cube' }, name: 'Large Cube' },
        ],
      }),
    ).toMatchObject({
      operations: [
        { type: 'entity.create' },
        { type: 'entity.setTransform' },
        { type: 'entity.rename' },
      ],
    });
  });

  it('rejects forward and duplicate tempId references before execution', () => {
    expect(() =>
      parseAgentEditorBatchRequest({
        editSessionId: 'edit-1',
        expectedSceneRevision: 4,
        operations: [
          { type: 'entity.rename', target: { tempId: 'cube' }, name: 'Too Early' },
          { type: 'entity.create', tempId: 'cube', kind: 'cube' },
          { type: 'entity.create', tempId: 'cube', kind: 'sphere' },
        ],
      }),
    ).toThrow(/tempId/);
  });

  it('rejects malformed operation payloads instead of accepting generic objects', () => {
    expect(() =>
      parseAgentEditorBatchRequest({
        editSessionId: 'edit-1',
        expectedSceneRevision: 4,
        operations: [
          {
            type: 'entity.setTransform',
            target: { guid: 'entity-guid' },
            scale: [4, 4, 4],
          },
        ],
      }),
    ).toThrow();
  });
});
