export type HostAssetReference = {
  guid: string;
  expectedType?: string;
  pathHint?: string;
};

export type AssetReferenceSource = {
  id: string;
  guid?: string;
  typeId?: string;
  path: string;
  sourcePath?: string;
  scope?: string;
};

export const hostAssetReference = (asset: AssetReferenceSource): HostAssetReference | null => {
  const guid = asset.guid || asset.id;
  if (!guid || asset.scope === 'procedural') return null;
  return {
    guid,
    ...(asset.typeId ? { expectedType: asset.typeId } : {}),
    pathHint: asset.sourcePath || asset.path,
  };
};
