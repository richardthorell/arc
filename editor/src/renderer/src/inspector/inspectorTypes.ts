import {
  aggregateInspectorSnapshots as aggregateBaseInspectorSnapshots,
  parseSelectedEntitySnapshot as parseBaseSelectedEntitySnapshot,
} from './inspectorTypesBase';
import type {
  InspectorEntitySnapshot as BaseInspectorEntitySnapshot,
  InspectorMeshRenderer as BaseInspectorMeshRenderer,
} from './inspectorTypesBase';

export * from './inspectorTypesBase';

export type InspectorMeshRenderer = BaseInspectorMeshRenderer & {
  // Mesh metadata was added after the original inspector snapshot contract. Keep
  // it optional for hand-authored/legacy snapshots; the parser normalizes it.
  hasMesh?: boolean;
  assetBackedMesh?: boolean;
  meshName?: string;
  meshPath?: string;
};

export type InspectorProceduralMesh = {
  type: 'plane' | 'cube' | 'sphere' | 'cylinder' | 'cone' | 'capsule';
  width?: number;
  height?: number;
  depth?: number;
  radius?: number;
  segmentsX?: number;
  segmentsY?: number;
  segmentsZ?: number;
  segments?: number;
  rings?: number;
  radialSegments?: number;
  hemisphereRings?: number;
  heightSegments?: number;
};

export type InspectorSkeletonJoint = {
  index: number;
  name: string;
  parent: number;
  bindPosition: [number, number, number];
  bindRotation: [number, number, number, number];
  bindScale: [number, number, number];
};

export type InspectorSkeleton = {
  name: string;
  selectedJoint: number;
  joints: InspectorSkeletonJoint[];
};

export type InspectorEntitySnapshot = Omit<BaseInspectorEntitySnapshot, 'meshRenderer'> & {
  meshRenderer: InspectorMeshRenderer | null;
  proceduralMesh?: InspectorProceduralMesh | null;
  skeleton?: InspectorSkeleton | null;
};

const finiteNumber = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? value : undefined);
const proceduralTypes = new Set<InspectorProceduralMesh['type']>([
  'plane',
  'cube',
  'sphere',
  'cylinder',
  'cone',
  'capsule',
]);

const tuple = (value: unknown, length: number): number[] | null => {
  if (!Array.isArray(value) || value.length !== length || value.some((entry) => typeof entry !== 'number')) return null;
  return value as number[];
};

function parseSkeleton(value: unknown): InspectorSkeleton | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Record<string, unknown>;
  if (!Array.isArray(raw.joints)) return null;
  const joints = raw.joints.flatMap((entry) => {
    if (!entry || typeof entry !== 'object') return [];
    const joint = entry as Record<string, unknown>;
    const bindPosition = tuple(joint.bindPosition, 3);
    const bindRotation = tuple(joint.bindRotation, 4);
    const bindScale = tuple(joint.bindScale, 3);
    if (
      typeof joint.index !== 'number' ||
      typeof joint.parent !== 'number' ||
      !bindPosition ||
      !bindRotation ||
      !bindScale
    )
      return [];
    return [
      {
        index: joint.index,
        name: typeof joint.name === 'string' ? joint.name : `Joint ${joint.index}`,
        parent: joint.parent,
        bindPosition: bindPosition as [number, number, number],
        bindRotation: bindRotation as [number, number, number, number],
        bindScale: bindScale as [number, number, number],
      },
    ];
  });
  if (!joints.length) return null;
  return {
    name: typeof raw.name === 'string' ? raw.name : 'Skeleton',
    selectedJoint: typeof raw.selectedJoint === 'number' ? raw.selectedJoint : joints[0].index,
    joints,
  };
}

function parseProceduralMesh(value: unknown): InspectorProceduralMesh | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Record<string, unknown>;
  if (typeof raw.type !== 'string' || !proceduralTypes.has(raw.type as InspectorProceduralMesh['type'])) return null;
  return {
    type: raw.type as InspectorProceduralMesh['type'],
    width: finiteNumber(raw.width),
    height: finiteNumber(raw.height),
    depth: finiteNumber(raw.depth),
    radius: finiteNumber(raw.radius),
    segmentsX: finiteNumber(raw.segmentsX),
    segmentsY: finiteNumber(raw.segmentsY),
    segmentsZ: finiteNumber(raw.segmentsZ),
    segments: finiteNumber(raw.segments),
    rings: finiteNumber(raw.rings),
    radialSegments: finiteNumber(raw.radialSegments),
    hemisphereRings: finiteNumber(raw.hemisphereRings),
    heightSegments: finiteNumber(raw.heightSegments),
  };
}

export function parseSelectedEntitySnapshot(value: unknown): InspectorEntitySnapshot {
  const parsed = parseBaseSelectedEntitySnapshot(value) as BaseInspectorEntitySnapshot;
  const raw = value && typeof value === 'object' ? (value as Record<string, unknown>) : undefined;
  const rawMeshRenderer =
    raw?.meshRenderer && typeof raw.meshRenderer === 'object'
      ? (raw.meshRenderer as Record<string, unknown>)
      : undefined;

  return {
    ...parsed,
    meshRenderer: parsed.meshRenderer
      ? {
          ...parsed.meshRenderer,
          hasMesh: rawMeshRenderer?.hasMesh === true,
          assetBackedMesh: rawMeshRenderer?.assetBackedMesh === true,
          meshName: typeof rawMeshRenderer?.meshName === 'string' ? rawMeshRenderer.meshName : '',
          meshPath: typeof rawMeshRenderer?.meshPath === 'string' ? rawMeshRenderer.meshPath : '',
        }
      : null,
    proceduralMesh: parseProceduralMesh(raw?.proceduralMesh),
    skeleton: parseSkeleton(raw?.skeleton),
  };
}

export function aggregateInspectorSnapshots(
  primary: InspectorEntitySnapshot,
  snapshots: ReadonlyArray<InspectorEntitySnapshot>,
): InspectorEntitySnapshot {
  // The base aggregator spreads the primary snapshot, so extension metadata is
  // preserved at runtime. Re-export it with the extended snapshot type as well.
  return aggregateBaseInspectorSnapshots(primary, snapshots) as InspectorEntitySnapshot;
}
