import type { ArcResourceHandler, ArcResourceMetadata } from './arcResourceRegistry';
import type { ArcUri } from './arcUri';

type HostResponse<T = unknown> = {
  succeeded: boolean;
  error?: string;
  payload?: T;
};

export type ArcAssetRecord = {
  guid: string;
  name?: string;
  title?: string;
  path: string;
  kind?: string;
  typeId?: string;
  scope?: string;
  state?: string;
  generation?: number;
};

type AssetInventory = { assets?: ArcAssetRecord[] };
type AssetThumbnail = { path: string; width: number; height: number; dataUrl: string };

export type ArcAssetResourceEnvironment = {
  listAssets: () => Promise<readonly ArcAssetRecord[]>;
  loadThumbnail: (path: string, maxSize: number) => Promise<AssetThumbnail | null>;
};

const normalizedId = (value: string) => value.trim().toLocaleLowerCase();
const titleCase = (value: string) =>
  value.replace(/(^|[-_])([a-z])/gu, (_, prefix, letter) => `${prefix ? ' ' : ''}${letter.toUpperCase()}`);

const thumbnailSize = (value: string | undefined): number | null => {
  if (value === undefined) return 128;
  if (!/^\d+$/u.test(value)) return null;
  const size = Number(value);
  return Number.isSafeInteger(size) && size >= 32 && size <= 512 ? size : null;
};

const resourceAsset = async (
  environment: ArcAssetResourceEnvironment,
  uri: ArcUri,
): Promise<ArcAssetRecord | null> => {
  const wanted = normalizedId(uri.id);
  return (await environment.listAssets()).find((asset) => normalizedId(asset.guid) === wanted) ?? null;
};

const metadataFor = (uri: ArcUri, asset: ArcAssetRecord): ArcResourceMetadata => ({
  uri,
  label: asset.title?.trim() || asset.name?.trim() || asset.path.split('/').pop() || asset.guid,
  subtitle: titleCase(asset.kind || asset.typeId || 'asset'),
  ...(asset.generation !== undefined ? { generation: asset.generation } : {}),
  metadata: {
    guid: asset.guid,
    path: asset.path,
    kind: asset.kind ?? asset.typeId ?? '',
    scope: asset.scope ?? '',
    state: asset.state ?? '',
  },
});

export const createArcAssetResourceHandler = (environment: ArcAssetResourceEnvironment): ArcResourceHandler => ({
  kind: 'asset',
  async resolve(uri) {
    if (uri.path.length > 1) return null;
    const asset = await resourceAsset(environment, uri);
    return asset ? metadataFor(uri, asset) : null;
  },
  async read(uri) {
    if (uri.path.length !== 1 || uri.path[0] !== 'thumbnail') return null;
    const size = thumbnailSize(uri.query.get('size'));
    if (size === null || [...uri.query.keys()].some((key) => key !== 'size')) return null;

    const asset = await resourceAsset(environment, uri);
    if (!asset) return null;
    const thumbnail = await environment.loadThumbnail(asset.path, size);
    if (!thumbnail?.dataUrl) return null;
    return {
      uri,
      mediaType: 'image/png',
      dataUrl: thumbnail.dataUrl,
      ...(asset.generation !== undefined ? { generation: asset.generation } : {}),
      metadata: { width: thumbnail.width, height: thumbnail.height, path: asset.path },
    };
  },
});

export const createWindowArcAssetResourceEnvironment = (): ArcAssetResourceEnvironment => ({
  async listAssets() {
    if (!window.arc?.host) return [];
    const response = (await window.arc.host.query('project.assets')) as HostResponse<AssetInventory>;
    return response.succeeded ? (response.payload?.assets ?? []).filter((asset) => Boolean(asset.guid && asset.path)) : [];
  },
  async loadThumbnail(path, maxSize) {
    if (!window.arc?.host) return null;
    const response = (await window.arc.host.query('asset.thumbnail', { path, maxSize })) as HostResponse<AssetThumbnail>;
    return response.succeeded ? (response.payload ?? null) : null;
  },
});
