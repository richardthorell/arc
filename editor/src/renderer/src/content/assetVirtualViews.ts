export type AssetVirtualViewKind = 'favorites' | 'recent' | 'downloads' | 'search-results';

export interface AssetVirtualViewAsset {
  assetId: string;
}

export interface AssetVirtualView {
  id: string;
  kind: AssetVirtualViewKind;
  assetIds: readonly string[];
  transient: boolean;
}

const persistentKinds = new Set<AssetVirtualViewKind>(['favorites', 'recent', 'downloads']);

export function createAssetVirtualView(kind: AssetVirtualViewKind, assetIds: readonly string[]): AssetVirtualView {
  const normalizedIds = normalizeAssetIds(assetIds);
  return {
    id: `virtual:${kind}`,
    kind,
    assetIds: normalizedIds,
    transient: !persistentKinds.has(kind),
  };
}

export function resolveAssetVirtualView<T extends AssetVirtualViewAsset>(
  view: AssetVirtualView,
  assets: readonly T[],
): T[] {
  const byId = new Map(assets.map((asset) => [asset.assetId, asset]));
  return view.assetIds.flatMap((assetId) => {
    const asset = byId.get(assetId);
    return asset === undefined ? [] : [asset];
  });
}

export function addAssetToVirtualView(view: AssetVirtualView, assetId: string): AssetVirtualView {
  const normalizedId = assetId.trim();
  if (normalizedId.length === 0 || view.assetIds.includes(normalizedId)) {
    return view;
  }

  return { ...view, assetIds: [...view.assetIds, normalizedId] };
}

export function removeAssetFromVirtualView(view: AssetVirtualView, assetId: string): AssetVirtualView {
  const nextIds = view.assetIds.filter((candidate) => candidate !== assetId);
  return nextIds.length === view.assetIds.length ? view : { ...view, assetIds: nextIds };
}

function normalizeAssetIds(assetIds: readonly string[]): string[] {
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const assetId of assetIds) {
    const candidate = assetId.trim();
    if (candidate.length === 0 || seen.has(candidate)) {
      continue;
    }
    seen.add(candidate);
    normalized.push(candidate);
  }
  return normalized;
}
