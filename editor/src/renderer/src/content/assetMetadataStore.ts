import type { AssetItem } from '../services/editorHostTypes';

export const assetMetadataFilePath = '.arc-asset-metadata.json';
export const assetMetadataSchemaVersion = 1;

export type AssetMetadata = {
  title?: string;
  description?: string;
  tags?: string[];
};

export type AssetMetadataEntries = Record<string, AssetMetadata>;

type AssetMetadataDocument = {
  version: number;
  assets: AssetMetadataEntries;
};

type AssetMetadataFileApi = {
  readText: (path: string, scope?: 'project' | 'builtin') => Promise<{ text: string }>;
  writeText: (path: string, text: string) => Promise<{ succeeded: boolean }>;
};

const normalizedPath = (value: string) => value.replaceAll('\\', '/').trim().toLocaleLowerCase();

export const assetMetadataKey = (asset: Pick<AssetItem, 'guid' | 'path'>): string =>
  asset.guid ? `guid:${asset.guid.toLocaleLowerCase()}` : `path:${normalizedPath(asset.path)}`;

const cleanOptionalText = (value: unknown): string | undefined => {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return trimmed || undefined;
};

export const normalizeAssetTags = (tags: readonly string[]): string[] => {
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const rawTag of tags) {
    const tag = rawTag.trim();
    const key = tag.toLocaleLowerCase();
    if (!tag || seen.has(key)) continue;
    seen.add(key);
    normalized.push(tag);
  }
  return normalized;
};

export const normalizeAssetMetadata = (metadata: AssetMetadata): AssetMetadata => {
  const title = cleanOptionalText(metadata.title);
  const description = cleanOptionalText(metadata.description);
  const tags = normalizeAssetTags(metadata.tags ?? []);
  return {
    ...(title ? { title } : {}),
    ...(description ? { description } : {}),
    ...(tags.length > 0 ? { tags } : {}),
  };
};

const isMetadataRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value && typeof value === 'object' && !Array.isArray(value));

export const parseAssetMetadata = (text: string): AssetMetadataEntries => {
  const parsed = JSON.parse(text) as unknown;
  if (!isMetadataRecord(parsed) || parsed.version !== assetMetadataSchemaVersion || !isMetadataRecord(parsed.assets)) {
    return {};
  }

  const entries: AssetMetadataEntries = {};
  for (const [key, value] of Object.entries(parsed.assets)) {
    if (!isMetadataRecord(value)) continue;
    const metadata = normalizeAssetMetadata({
      title: value.title as string | undefined,
      description: value.description as string | undefined,
      tags: Array.isArray(value.tags) ? value.tags.filter((tag): tag is string => typeof tag === 'string') : undefined,
    });
    if (Object.keys(metadata).length > 0) entries[key] = metadata;
  }
  return entries;
};

export const serializeAssetMetadata = (entries: AssetMetadataEntries): string => {
  const assets = Object.fromEntries(
    Object.entries(entries)
      .map(([key, value]) => [key, normalizeAssetMetadata(value)] as const)
      .filter(([, value]) => Object.keys(value).length > 0)
      .sort(([left], [right]) => left.localeCompare(right)),
  );
  const document: AssetMetadataDocument = { version: assetMetadataSchemaVersion, assets };
  return `${JSON.stringify(document, null, 2)}\n`;
};

export const applyAssetMetadata = (assets: readonly AssetItem[], entries: AssetMetadataEntries): AssetItem[] =>
  assets.map((asset) => {
    const metadata = entries[assetMetadataKey(asset)];
    return metadata ? { ...asset, ...metadata } : asset;
  });

export const setAssetMetadataEntry = (
  entries: AssetMetadataEntries,
  asset: Pick<AssetItem, 'guid' | 'path'>,
  metadata: AssetMetadata,
): AssetMetadataEntries => {
  const key = assetMetadataKey(asset);
  const normalized = normalizeAssetMetadata(metadata);
  const next = { ...entries };
  if (Object.keys(normalized).length === 0) delete next[key];
  else next[key] = normalized;
  return next;
};

export const loadAssetMetadata = async (api: AssetMetadataFileApi = window.arc.projects): Promise<AssetMetadataEntries> => {
  try {
    const file = await api.readText(assetMetadataFilePath);
    return parseAssetMetadata(file.text);
  } catch {
    return {};
  }
};

export const saveAssetMetadata = async (
  entries: AssetMetadataEntries,
  api: AssetMetadataFileApi = window.arc.projects,
): Promise<void> => {
  const result = await api.writeText(assetMetadataFilePath, serializeAssetMetadata(entries));
  if (!result.succeeded) throw new Error('Could not save asset metadata');
};
