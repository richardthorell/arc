import { describe, expect, it } from 'vitest';

import type { SceneEntity } from '../services/editorHostTypes';
import type { InspectorTransform } from './inspectorTypes';
import { inspectorLocalToWorld, inspectorWorldToLocal } from './transformSpace';

const transform = (positionX: number, scale = 1, rotationY = 0): InspectorTransform => ({
  position: { x: positionX, y: 0, z: 0 },
  rotationDegrees: { x: 0, y: rotationY, z: 0 },
  scale: { x: scale, y: scale, z: scale },
  rotationQuaternion: { x: 0, y: 0, z: 0, w: 1 },
});

const scene: SceneEntity[] = [
  {
    id: '1:0',
    name: 'Parent',
    kind: 'folder',
    active: true,
    transform: { position: { x: 10, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: 0 }, scale: { x: 2, y: 2, z: 2 } },
    children: [
      {
        id: '2:0',
        parentId: '1:0',
        name: 'Child',
        kind: 'mesh',
        active: true,
        transform: { position: { x: 1, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: 0 }, scale: { x: 1, y: 1, z: 1 } },
      },
    ],
  },
];

describe('transformSpace', () => {
  it('composes parent translation and scale for world-space display', () => {
    const world = inspectorLocalToWorld(scene, '2:0', transform(1));
    expect(world.position.x).toBeCloseTo(12);
    expect(world.scale.x).toBeCloseTo(2);
  });

  it('round-trips world edits back to local TRS', () => {
    const local = transform(1, 0.75, 37);
    const world = inspectorLocalToWorld(scene, '2:0', local);
    const roundTrip = inspectorWorldToLocal(scene, '2:0', world);
    expect(roundTrip.position.x).toBeCloseTo(local.position.x, 5);
    expect(roundTrip.scale.x).toBeCloseTo(local.scale.x, 5);
    expect(roundTrip.rotationDegrees.y).toBeCloseTo(local.rotationDegrees.y, 5);
  });
});
