import { describe, expect, it } from 'vitest';

import type { SceneEntity } from '../services/editorHostTypes';
import { cameraTargetFromTransform, collectSceneCameraSources } from './viewportCameraSource';

const camera = (rotation = { x: 0, y: 0, z: 0 }): SceneEntity => ({
  id: 'camera-1',
  name: 'Gameplay Camera',
  kind: 'camera',
  active: true,
  transform: {
    position: { x: 1, y: 2, z: 3 },
    rotation,
    scale: { x: 1, y: 1, z: 1 },
  },
});

describe('viewport camera sources', () => {
  it('converts an unrotated scene camera into a -Z viewport target', () => {
    expect(cameraTargetFromTransform(camera())).toEqual({
      id: 'camera-1',
      name: 'Gameplay Camera',
      position: [1, 2, 3],
      target: [1, 2, -7],
    });
  });

  it('collects active scene cameras recursively and skips inactive cameras', () => {
    const inactive = { ...camera(), id: 'inactive', active: false };
    const root: SceneEntity = {
      id: 'folder',
      name: 'Cameras',
      kind: 'folder',
      active: true,
      children: [camera({ x: 0, y: 90, z: 0 }), inactive],
    };
    const sources = collectSceneCameraSources([root]);
    expect(sources).toHaveLength(1);
    expect(sources[0]?.id).toBe('camera-1');
    expect(sources[0]?.target[0]).toBeCloseTo(-9);
  });
});
