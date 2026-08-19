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

export type InspectorEntitySnapshot = Omit<BaseInspectorEntitySnapshot, 'meshRenderer'> & {
  meshRenderer: InspectorMeshRenderer | null;
  proceduralMesh: InspectorProceduralMesh | null;
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
  const rawMeshRenderer = raw?.meshRenderer && typeof raw.meshRenderer === 'object'
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
