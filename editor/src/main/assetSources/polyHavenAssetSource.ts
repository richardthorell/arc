import type {
  ArcAssetDownloadFile,
  ArcAssetDownloadManifest,
  ArcAssetSearchResult,
  ArcAssetSourceAdapter,
  ArcAssetSourceDescriptor,
  ArcAssetSourceQuery,
  ArcRemoteAsset,
  ArcRemoteAssetKind,
} from '../../common/assetSourceTypes';

type JsonFetcher = (url: string, headers: Record<string, string>) => Promise<unknown>;

type PolyHavenAsset = {
  name?: unknown;
  description?: unknown;
  category?: unknown;
  tags?: unknown;
  attributes?: unknown;
  thumbnail_url?: unknown;
  max_resolution?: unknown;
  dimensions?: unknown;
  polycount?: unknown;
  texel_density?: unknown;
  download_count?: unknown;
  authors?: unknown;
  date_published?: unknown;
  files_hash?: unknown;
  type?: unknown;
  [key: string]: unknown;
};

type PolyHavenCatalog = Record<string, PolyHavenAsset>;

export type PolyHavenAssetSourceOptions = {
  baseUrl?: string;
  userAgent?: string;
  cacheDurationMs?: number;
  fetchJson?: JsonFetcher;
};

const sourceId = 'polyhaven';
const defaultBaseUrl = 'https://api.polyhaven.com';
const defaultCacheDurationMs = 5 * 60 * 1000;

const descriptor: ArcAssetSourceDescriptor = {
  id: sourceId,
  displayName: 'Poly Haven',
  homepage: 'https://polyhaven.com',
  attribution: 'Powered by Poly Haven',
  licenseSummary: 'CC0',
  capabilities: {
    search: true,
    downloadManifest: true,
  },
};

const asString = (value: unknown): string => (typeof value === 'string' ? value : '');
const asNumber = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined;
const asStringArray = (value: unknown): string[] =>
  Array.isArray(value)
    ? value
        .filter((entry): entry is string => typeof entry === 'string')
        .map((entry) => entry.trim())
        .filter(Boolean)
    : [];

const assetKind = (value: unknown): ArcRemoteAssetKind => {
  if (value === 0) return 'hdri';
  if (value === 1) return 'texture';
  if (value === 2) return 'model';
  return 'other';
};

const publishedAt = (value: unknown): string | undefined => {
  const seconds = asNumber(value);
  if (seconds === undefined) return undefined;
  const date = new Date(seconds * 1000);
  return Number.isNaN(date.getTime()) ? undefined : date.toISOString();
};

const normalizeAsset = (id: string, raw: PolyHavenAsset): ArcRemoteAsset => ({
  id,
  sourceId,
  name: asString(raw.name) || id,
  description: asString(raw.description),
  kind: assetKind(raw.type),
  category: asString(raw.category),
  tags: asStringArray(raw.tags),
  thumbnailUrl: asString(raw.thumbnail_url) || undefined,
  license: 'CC0',
  attribution: descriptor.attribution,
  publishedAt: publishedAt(raw.date_published),
  metadata: {
    attributes: raw.attributes ?? {},
    authors: raw.authors ?? {},
    maxResolution: raw.max_resolution,
    dimensions: raw.dimensions,
    polycount: raw.polycount,
    texelDensity: raw.texel_density,
    downloadCount: raw.download_count,
    filesHash: raw.files_hash,
  },
});

const defaultFetcher: JsonFetcher = async (url, headers) => {
  const response = await fetch(url, { headers });
  if (!response.ok) throw new Error(`Poly Haven request failed (${response.status} ${response.statusText})`);
  return response.json();
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const validateCatalog = (value: unknown): PolyHavenCatalog => {
  if (!isRecord(value)) throw new Error('Poly Haven returned an invalid asset catalog');
  return value as PolyHavenCatalog;
};

const flattenFiles = (
  value: unknown,
  path: string[] = [],
  output: ArcAssetDownloadFile[] = [],
): ArcAssetDownloadFile[] => {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => flattenFiles(entry, [...path, String(index)], output));
    return output;
  }
  if (!isRecord(value)) return output;

  if (typeof value.url === 'string') {
    const size = asNumber(value.size);
    const md5 = asString(value.md5);
    const sha256 = asString(value.sha256);
    output.push({
      logicalPath: path.join('/'),
      url: value.url,
      sizeBytes: size,
      checksum: sha256 ? { algorithm: 'sha256', value: sha256 } : md5 ? { algorithm: 'md5', value: md5 } : undefined,
    });
  }

  for (const [key, child] of Object.entries(value)) {
    if (key === 'url' || key === 'size' || key === 'md5' || key === 'sha256') continue;
    flattenFiles(child, [...path, key], output);
  }
  return output;
};

export class PolyHavenAssetSource implements ArcAssetSourceAdapter {
  readonly descriptor = descriptor;

  private readonly baseUrl: string;
  private readonly userAgent: string;
  private readonly cacheDurationMs: number;
  private readonly fetchJson: JsonFetcher;
  private catalogCache: { value: PolyHavenCatalog; expiresAt: number } | null = null;

  constructor(options: PolyHavenAssetSourceOptions = {}) {
    this.baseUrl = (options.baseUrl ?? defaultBaseUrl).replace(/\/$/, '');
    this.userAgent = options.userAgent ?? 'ARC-Editor/0.1';
    this.cacheDurationMs = options.cacheDurationMs ?? defaultCacheDurationMs;
    this.fetchJson = options.fetchJson ?? defaultFetcher;
  }

  private headers(): Record<string, string> {
    return {
      Accept: 'application/json',
      'User-Agent': this.userAgent,
    };
  }

  private async catalog(): Promise<PolyHavenCatalog> {
    const now = Date.now();
    if (this.catalogCache && this.catalogCache.expiresAt > now) return this.catalogCache.value;
    const value = validateCatalog(await this.fetchJson(`${this.baseUrl}/assets`, this.headers()));
    this.catalogCache = { value, expiresAt: now + this.cacheDurationMs };
    return value;
  }

  async search(query: ArcAssetSourceQuery = {}): Promise<ArcAssetSearchResult> {
    const catalog = await this.catalog();
    const needle = query.text?.trim().toLocaleLowerCase() ?? '';
    const kinds = new Set(query.kinds ?? []);
    const limit = Math.max(1, Math.min(query.limit ?? 100, 500));

    const matches = Object.entries(catalog)
      .map(([id, raw]) => normalizeAsset(id, raw))
      .filter((asset) => kinds.size === 0 || kinds.has(asset.kind))
      .filter((asset) => {
        if (!needle) return true;
        return `${asset.name} ${asset.description} ${asset.category} ${asset.tags.join(' ')}`
          .toLocaleLowerCase()
          .includes(needle);
      })
      .sort((left, right) => left.name.localeCompare(right.name));

    return {
      source: this.descriptor,
      assets: matches.slice(0, limit),
      total: matches.length,
    };
  }

  async getAsset(assetId: string): Promise<ArcRemoteAsset | null> {
    const catalog = await this.catalog();
    const asset = catalog[assetId];
    return asset ? normalizeAsset(assetId, asset) : null;
  }

  async getDownloadManifest(assetId: string): Promise<ArcAssetDownloadManifest> {
    if (!assetId.trim()) throw new Error('Poly Haven asset id is required');
    const payload = await this.fetchJson(`${this.baseUrl}/files/${encodeURIComponent(assetId)}`, this.headers());
    const files = flattenFiles(payload);
    if (files.length === 0) throw new Error(`Poly Haven returned no downloadable files for '${assetId}'`);
    return {
      sourceId,
      assetId,
      files,
    };
  }
}
