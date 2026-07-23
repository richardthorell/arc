import { describe, expect, it } from 'vitest';

import {
  eulerDegreesToQuaternion,
  parseSelectedEntitySnapshot,
  quaternionToEulerDegrees,
  transformHostPayload,
} from './inspectorTypes';

describe('inspector host bindings', () => {
  it('parses the typed selected entity snapshot', () => {
    const snapshot = parseSelectedEntitySnapshot({
      entity: { index: 4, generation: 2 },
      name: 'Main Camera',
      tag: 'Camera',
      active: true,
      renderLayerMask: 2,
      transform: { position: [1, 2, 3], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
      camera: {
        projection: 'perspective', fovYDegrees: 60, orthographicHeight: 10,
        nearPlane: 0.1, farPlane: 2000, active: true, clearColor: [0.1, 0.2, 0.3, 1],
      },
      meshRenderer: null,
      components: [
        { kind: 'transform', label: 'Transform', editable: true },
        { kind: 'camera', label: 'Camera', editable: true },
      ],
    });

    expect(snapshot.entity).toEqual({ index: 4, generation: 2 });
    expect(snapshot.transform?.position).toEqual({ x: 1, y: 2, z: 3 });
    expect(snapshot.transform?.rotationDegrees.x).toBeCloseTo(0);
    expect(snapshot.camera?.clearColor).toEqual({ x: 0.1, y: 0.2, z: 0.3, w: 1 });
  });

  it('parses mesh renderer material bindings and linear tint', () => {
    const snapshot = parseSelectedEntitySnapshot({
      entity: { index: 8, generation: 1 }, name: 'Rock', tag: 'Mesh', active: true, renderLayerMask: 1,
      transform: null, camera: null,
      meshRenderer: {
        visible: true, baseColorTint: [0.8, 0.9, 1, 1], hasMaterial: true,
        assetBackedMaterial: true, materialName: 'Rock', materialPath: 'materials/rock.arcmat',
      },
      components: [{ kind: 'meshRenderer', label: 'Mesh Renderer', editable: true }],
    });
    expect(snapshot.meshRenderer?.materialPath).toBe('materials/rock.arcmat');
    expect(snapshot.meshRenderer?.baseColorTint).toEqual({ x: 0.8, y: 0.9, z: 1, w: 1 });
  });

  it('parses prefab instance source and override state', () => {
    const snapshot = parseSelectedEntitySnapshot({
      entity: { index: 9, generation: 4 }, name: 'Cabin', tag: 'Mesh', active: true, renderLayerMask: 1,
      transform: null, camera: null, meshRenderer: null, terrain: null,
      prefab: {
        prefabGuid: '1020304050607080a0b0c0d0e0f00102',
        prefabPath: 'assets/prefabs/cabin.arcprefab',
        overrideCount: 3,
        sourceMissing: false,
      },
      components: [{ kind: 'prefabInstance', label: 'Prefab Instance', editable: true }],
    });
    expect(snapshot.prefab?.prefabPath).toBe('assets/prefabs/cabin.arcprefab');
    expect(snapshot.prefab?.overrideCount).toBe(3);
  });

  it('round trips ARC XYZ Euler rotations through a quaternion', () => {
    const expected = { x: 21, y: -34, z: 67 };
    const quaternion = eulerDegreesToQuaternion(expected);
    const actual = quaternionToEulerDegrees(quaternion);
    expect(actual.x).toBeCloseTo(expected.x, 5);
    expect(actual.y).toBeCloseTo(expected.y, 5);
    expect(actual.z).toBeCloseTo(expected.z, 5);

    const payload = transformHostPayload({
      position: { x: 1, y: 2, z: 3 },
      rotationDegrees: expected,
      rotationQuaternion: quaternion,
      scale: { x: 2, y: 3, z: 4 },
    });
    expect(payload.position).toEqual([1, 2, 3]);
    expect(payload.scale).toEqual([2, 3, 4]);
    expect(payload.rotation).toHaveLength(4);
  });
});
