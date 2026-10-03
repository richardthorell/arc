import type { AssetItem } from '../services/editorHostTypes';
import { deserializeAssetVirtualViews, serializeAssetVirtualViews } from './assetVirtualViewPersistence';
import {
  addAssetToVirtualView,
  createAssetVirtualView,
  removeAssetFromVirtualView,
  resolveAssetVirtualView,
  type AssetVirtualView,
  type AssetVirtualViewKind,
} from './assetVirtualViews';

const durableKinds = ['favorites', 'recent', 'downloads'] as const satisfies readonly AssetVirtualViewKind[];
const legacyFavoritesStorageKey = 'arc.content.favorites';

export type AssetVirtualViewStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

export const assetVirtualViewStorageKey = (projectRoot: string) => `arc.content.virtualViews.v1:${projectRoot}`;

export const defaultAssetVirtualViews = (): AssetVirtualView[] =>
  durableKinds.map((kind) => createAssetVirtualView(kind, []));

export const isAssetVirtualViewKind = (value: string): value is AssetVirtualViewKind =>
  value === 'favorites' || value === 'recent' || value === 'downloads' || value === 'search-results';

export const normalizeAssetVirtualViews = (views: readonly AssetVirtualView[]): AssetVirtualView[] =>
  durableKinds.map((kind) => views.find((view) => view.kind === kind) ?? createAssetVirtualView(kind, []));

export function loadAssetVirtualViews(
  storage: AssetVirtualViewStorage,
  projectRoot: string,
  assets: readonly Pick<AssetItem, 'id' | 'guid' | 'path'>[],
): AssetVirtualView[] {
  const key = assetVirtualViewStorageKey(projectRoot);
  const serialized = storage.getItem(key);
  if (serialized !== null) {
    return normalizeAssetVirtualViews(deserializeAssetVirtualViews(serialized));
  }

  const views = defaultAssetVirtualViews();
  const legacy = parseLegacyFavorites(storage.getItem(legacyFavoritesStorageKey));
  if (legacy.length === 0) return views;

  const stableIds = legacy.flatMap((legacyId) => {
    const asset = assets.find(
      (candidate) => candidate.id === legacyId || candidate.guid === legacyId || candidate.path === legacyId,
    );
    return asset ? [asset.id] : [];
  });
  const migrated = replaceAssetVirtualView(views, createAssetVirtualView('favorites', stableIds));
  storage.setItem(key, serializeAssetVirtualViews(migrated));
  storage.removeItem(legacyFavoritesStorageKey);
  return migrated;
}

export const saveAssetVirtualViews = (
  storage: AssetVirtualViewStorage,
  projectRoot: string,
  views: readonly AssetVirtualView[],
) => storage.setItem(assetVirtualViewStorageKey(projectRoot), serializeAssetVirtualViews(normalizeAssetVirtualViews(views)));

export const assetVirtualViewForKind = (
  views: readonly AssetVirtualView[],
  kind: AssetVirtualViewKind,
  searchResultIds: readonly string[] = [],
): AssetVirtualView =>
  kind === 'search-results'
    ? createAssetVirtualView('search-results', searchResultIds)
    : (views.find((view) => view.kind === kind) ?? createAssetVirtualView(kind, []));

export const assetsForVirtualView = <T extends Pick<AssetItem, 'id'>>(
  views: readonly AssetVirtualView[],
  kind: AssetVirtualViewKind,
  assets: readonly T[],
  searchResultIds: readonly string[] = [],
): T[] => {
  const view = assetVirtualViewForKind(views, kind, searchResultIds);
  return resolveAssetVirtualView(
    view,
    assets.map((asset) => ({ assetId: asset.id, asset })),
  ).map(({ asset }) => asset);
};

export const assetVirtualViewContains = (
  views: readonly AssetVirtualView[],
  kind: AssetVirtualViewKind,
  assetId: string,
): boolean => assetVirtualViewForKind(views, kind).assetIds.includes(assetId);

export const setFavoriteAsset = (
  views: readonly AssetVirtualView[],
  assetId: string,
  favorite: boolean,
): AssetVirtualView[] => {
  const current = assetVirtualViewForKind(views, 'favorites');
  const next = favorite ? addAssetToVirtualView(current, assetId) : removeAssetFromVirtualView(current, assetId);
  return replaceAssetVirtualView(views, next);
};

export const recordRecentAsset = (
  views: readonly AssetVirtualView[],
  assetId: string,
  limit = 50,
): AssetVirtualView[] => {
  const current = assetVirtualViewForKind(views, 'recent');
  const ids = [assetId, ...current.assetIds.filter((candidate) => candidate !== assetId)].filter(Boolean).slice(0, limit);
  return replaceAssetVirtualView(views, createAssetVirtualView('recent', ids));
};

export const recordDownloadedAssets = (
  views: readonly AssetVirtualView[],
  assetIds: readonly string[],
): AssetVirtualView[] => {
  const current = assetVirtualViewForKind(views, 'downloads');
  const ids = [...assetIds, ...current.assetIds.filter((candidate) => !assetIds.includes(candidate))];
  return replaceAssetVirtualView(views, createAssetVirtualView('downloads', ids));
};

export const removeAssetFromVirtualCollection = (
  views: readonly AssetVirtualView[],
  kind: 'favorites' | 'downloads',
  assetId: string,
): AssetVirtualView[] =>
  replaceAssetVirtualView(views, removeAssetFromVirtualView(assetVirtualViewForKind(views, kind), assetId));

export const assetIdsMatchingImportedPaths = (
  assets: readonly Pick<AssetItem, 'id' | 'path' | 'sourcePath'>[],
  importedPaths: readonly string[],
): string[] => {
  const paths = importedPaths.map(normalizePath).filter(Boolean);
  if (paths.length === 0) return [];

  return [
    ...new Set(
      assets.flatMap((asset) => {
        const candidates = [asset.path, asset.sourcePath].filter((value): value is string => Boolean(value)).map(normalizePath);
        const matches = candidates.some((candidate) =>
          paths.some(
            (imported) =>
              candidate === imported || candidate.endsWith(`/${imported}`) || imported.endsWith(`/${candidate}`),
          ),
        );
        return matches ? [asset.id] : [];
      }),
    ),
  ];
};

const replaceAssetVirtualView = (views: readonly AssetVirtualView[], replacement: AssetVirtualView): AssetVirtualView[] => {
  const normalized = normalizeAssetVirtualViews(views);
  const index = normalized.findIndex((view) => view.kind === replacement.kind);
  if (index < 0) return [...normalized, replacement];
  return normalized.map((view, candidateIndex) => (candidateIndex === index ? replacement : view));
};

const parseLegacyFavorites = (serialized: string | null): string[] => {
  if (!serialized) return [];
  try {
    const parsed = JSON.parse(serialized) as unknown;
    return Array.isArray(parsed) ? parsed.filter((value): value is string => typeof value === 'string') : [];
  } catch {
    return [];
  }
};

const normalizePath = (value: string) => value.replaceAll('\\', '/').replace(/\/+/g, '/').replace(/^\/|\/$/g, '').toLowerCase();
