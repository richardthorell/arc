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

export type InspectorEntitySnapshot = Omit<BaseInspectorEntitySnapshot, 'meshRenderer'> & {
  meshRenderer: InspectorMeshRenderer | null;
};

export function parseSelectedEntitySnapshot(value: unknown): InspectorEntitySnapshot {
  const parsed = parseBaseSelectedEntitySnapshot(value) as BaseInspectorEntitySnapshot;
  if (!parsed.meshRenderer) return parsed as InspectorEntitySnapshot;

  const rawMeshRenderer =
    value && typeof value === 'object'
      ? (value as { meshRenderer?: Record<string, unknown> }).meshRenderer
      : undefined;

  return {
    ...parsed,
    meshRenderer: {
      ...parsed.meshRenderer,
      hasMesh: rawMeshRenderer?.hasMesh === true,
      assetBackedMesh: rawMeshRenderer?.assetBackedMesh === true,
      meshName: typeof rawMeshRenderer?.meshName === 'string' ? rawMeshRenderer.meshName : '',
      meshPath: typeof rawMeshRenderer?.meshPath === 'string' ? rawMeshRenderer.meshPath : '',
    },
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
