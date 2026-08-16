import { parseSelectedEntitySnapshot as parseBaseSelectedEntitySnapshot } from './inspectorTypesBase';
import type {
  InspectorEntitySnapshot as BaseInspectorEntitySnapshot,
  InspectorMeshRenderer as BaseInspectorMeshRenderer,
} from './inspectorTypesBase';

export * from './inspectorTypesBase';

export type InspectorMeshRenderer = BaseInspectorMeshRenderer & {
  hasMesh: boolean;
  assetBackedMesh: boolean;
  meshName: string;
  meshPath: string;
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
