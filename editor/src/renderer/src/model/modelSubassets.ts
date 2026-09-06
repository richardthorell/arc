import type { AssetItem } from '../services/editorHostTypes';

export type ModelSubasset = {
  id: string;
  kind: 'mesh' | 'skeleton';
  name: string;
  meshIndex?: number;
};

const modelBaseName = (asset: Pick<AssetItem, 'name' | 'path'>) => {
  const raw = asset.name || asset.path.replaceAll('\\\\', '/').split('/').at(-1) || 'Model';
  return raw.replace(/\.(fbx|glb|gltf|obj)$/i, '');
};

export function hasSkeletonMetadata(asset: AssetItem) {
  return Boolean((asset.skeletonJoints?.length ?? 0) > 0 || (asset.skeletonBoneCount ?? 0) > 0);
}

export function buildModelSubassets(asset: AssetItem): ModelSubasset[] {
  const result: ModelSubasset[] = [];
  const meshCount = Math.max(0, Math.round(asset.meshCount ?? (asset.kind === 'mesh' ? 1 : 0)));
  for (let index = 0; index < meshCount; index += 1) {
    result.push({
      id: `${asset.guid ?? asset.path}:mesh:${index}`,
      kind: 'mesh',
      name: meshCount === 1 ? 'Mesh' : `Mesh ${index + 1}`,
      meshIndex: index,
    });
  }
  if (hasSkeletonMetadata(asset)) {
    result.push({
      id: `${asset.guid ?? asset.path}:skeleton`,
      kind: 'skeleton',
      name: asset.skeletonName?.trim() || `${modelBaseName(asset)} Skeleton`,
    });
  }
  return result;
}

export type SkeletonCompatibility = 'compatible' | 'partial' | 'incompatible';

export function skeletonCompatibility(source: AssetItem, candidate: AssetItem): SkeletonCompatibility {
  const sourceJoints = source.skeletonJoints ?? [];
  const candidateJoints = candidate.skeletonJoints ?? [];
  const sourceCount = source.skeletonBoneCount ?? sourceJoints.length;
  const candidateCount = candidate.skeletonBoneCount ?? candidateJoints.length;
  if (!sourceCount || !candidateCount || sourceCount !== candidateCount) return 'incompatible';
  if (sourceJoints.length && candidateJoints.length) {
    const sourceNames = sourceJoints.map((joint) => joint.name);
    const candidateNames = candidateJoints.map((joint) => joint.name);
    return sourceNames.every((name, index) => name === candidateNames[index]) ? 'compatible' : 'partial';
  }
  return 'partial';
}

export function skeletonHierarchyDepth(asset: AssetItem) {
  if (asset.skeletonHierarchyDepth !== undefined) return asset.skeletonHierarchyDepth;
  const joints = asset.skeletonJoints ?? [];
  if (!joints.length) return 0;
  const byIndex = new Map(joints.map((joint) => [joint.index, joint]));
  let depth = 0;
  for (const joint of joints) {
    let current = joint;
    let currentDepth = 1;
    const seen = new Set<number>();
    while (current.parent >= 0 && !seen.has(current.index)) {
      seen.add(current.index);
      const parent = byIndex.get(current.parent);
      if (!parent) break;
      current = parent;
      currentDepth += 1;
    }
    depth = Math.max(depth, currentDepth);
  }
  return depth;
}
