import { AssetVirtualView, AssetVirtualViewKind, createAssetVirtualView } from './assetVirtualViews';

const schemaVersion = 1;
const persistentKinds = new Set<AssetVirtualViewKind>(['favorites', 'recent', 'downloads']);

interface PersistedAssetVirtualViewsV1 {
  version: 1;
  views: Array<{
    kind: AssetVirtualViewKind;
    assetIds: string[];
  }>;
}

export function serializeAssetVirtualViews(views: readonly AssetVirtualView[]): string {
  const durableViews = views
    .filter((view) => persistentKinds.has(view.kind) && !view.transient)
    .map((view) => ({ kind: view.kind, assetIds: [...view.assetIds] }));

  const payload: PersistedAssetVirtualViewsV1 = {
    version: schemaVersion,
    views: durableViews,
  };
  return JSON.stringify(payload);
}

export function deserializeAssetVirtualViews(serialized: string): AssetVirtualView[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    return [];
  }

  if (!isRecord(parsed) || parsed.version !== schemaVersion || !Array.isArray(parsed.views)) {
    return [];
  }

  const views: AssetVirtualView[] = [];
  const seenKinds = new Set<AssetVirtualViewKind>();
  for (const candidate of parsed.views) {
    if (!isRecord(candidate) || !isPersistentKind(candidate.kind) || !isStringArray(candidate.assetIds)) {
      continue;
    }
    if (seenKinds.has(candidate.kind)) {
      continue;
    }
    seenKinds.add(candidate.kind);
    views.push(createAssetVirtualView(candidate.kind, candidate.assetIds));
  }
  return views;
}

function isPersistentKind(value: unknown): value is AssetVirtualViewKind {
  return typeof value === 'string' && persistentKinds.has(value as AssetVirtualViewKind);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === 'string');
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
