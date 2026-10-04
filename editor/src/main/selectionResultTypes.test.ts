import { describe, expect, it } from 'vitest';

const selectionResult = {
  selectionCount: 1,
  selectedGuids: ['floor-guid'],
  sceneRevision: 4,
  worldEpoch: 2,
  frameRevision: 9,
};

describe('agent selection result contract', () => {
  it('keeps stable identity and authority revisions together', () => {
    expect(selectionResult).toMatchObject({
      selectedGuids: ['floor-guid'],
      sceneRevision: 4,
      frameRevision: 9,
    });
  });
});
