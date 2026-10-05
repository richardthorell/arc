import { describe, expect, it } from 'vitest';

import { parseAgentEditorBatchRequest } from './agentEditorBatch';

describe('agent editor batch contract', () => {
  it('accepts entity creation followed by tempId-based edits', () => {
    const parsed = parseAgentEditorBatchRequest({
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
    });

    expect(parsed.operations.map((operation) => operation.type)).toEqual([
      'entity.create',
      'entity.setTransform',
      'entity.rename',
    ]);
  });

  it('accepts a base-color override without creating a new material asset', () => {
    const parsed = parseAgentEditorBatchRequest({
      editSessionId: 'edit-1',
      expectedSceneRevision: 4,
      operations: [
        { type: 'entity.create', tempId: 'capsule', kind: 'capsule' },
        { type: 'entity.rename', target: { tempId: 'capsule' }, name: 'Green Capsule' },
        {
          type: 'entity.setTransform',
          target: { tempId: 'capsule' },
          transform: {
            position: [0, 0, 0],
            rotation: [0, 0, 0, 1],
            scale: [5, 5, 5],
          },
        },
        { type: 'entity.setBaseColor', target: { tempId: 'capsule' }, color: [0.1, 0.8, 0.1] },
      ],
    });

    expect(parsed.operations.map((operation) => operation.type)).toEqual([
      'entity.create',
      'entity.rename',
      'entity.setTransform',
      'entity.setBaseColor',
    ]);
  });

  it('accepts material creation and assignment through a material tempId', () => {
    const parsed = parseAgentEditorBatchRequest({
      editSessionId: 'edit-1',
      expectedSceneRevision: 4,
      operations: [
        { type: 'entity.create', tempId: 'capsule', kind: 'capsule' },
        { type: 'entity.rename', target: { tempId: 'capsule' }, name: 'Red Capsule' },
        {
          type: 'entity.setTransform',
          target: { tempId: 'capsule' },
          transform: {
            position: [0, 0, 0],
            rotation: [0, 0, 0, 1],
            scale: [5, 5, 5],
          },
        },
        {
          type: 'material.create',
          tempId: 'redMaterial',
          path: 'materials/red_capsule.arcmat',
          baseColor: [1, 0, 0, 1],
        },
        {
          type: 'entity.setMaterial',
          target: { tempId: 'capsule' },
          material: { tempId: 'redMaterial' },
        },
      ],
    });

    expect(parsed.operations.map((operation) => operation.type)).toEqual([
      'entity.create',
      'entity.rename',
      'entity.setTransform',
      'material.create',
      'entity.setMaterial',
    ]);
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

    expect(() =>
      parseAgentEditorBatchRequest({
        editSessionId: 'edit-1',
        expectedSceneRevision: 4,
        operations: [
          { type: 'entity.create', tempId: 'capsule', kind: 'capsule' },
          {
            type: 'entity.setMaterial',
            target: { tempId: 'capsule' },
            material: { tempId: 'redMaterial' },
          },
          {
            type: 'material.create',
            tempId: 'redMaterial',
            path: 'materials/red_capsule.arcmat',
            baseColor: [1, 0, 0, 1],
          },
        ],
      }),
    ).toThrow(/material created earlier/);
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
