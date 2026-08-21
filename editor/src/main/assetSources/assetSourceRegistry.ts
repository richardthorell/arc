import type {
  ArcAssetDownloadManifest,
  ArcAssetSearchResult,
  ArcAssetSourceAdapter,
  ArcAssetSourceDescriptor,
  ArcAssetSourceQuery,
  ArcRemoteAsset,
} from '../../common/assetSourceTypes';
import { PolyHavenAssetSource } from './polyHavenAssetSource';

export class AssetSourceRegistry {
  private readonly sources = new Map<string, ArcAssetSourceAdapter>();

  constructor(sources: ArcAssetSourceAdapter[] = []) {
    sources.forEach((source) => this.register(source));
  }

  register(source: ArcAssetSourceAdapter): void {
    if (this.sources.has(source.descriptor.id))
      throw new Error(`Asset source '${source.descriptor.id}' is already registered`);
    this.sources.set(source.descriptor.id, source);
  }

  list(): ArcAssetSourceDescriptor[] {
    return [...this.sources.values()].map((source) => source.descriptor);
  }

  private require(sourceId: string): ArcAssetSourceAdapter {
    const source = this.sources.get(sourceId);
    if (!source) throw new Error(`Unknown asset source '${sourceId}'`);
    return source;
  }

  search(sourceId: string, query?: ArcAssetSourceQuery): Promise<ArcAssetSearchResult> {
    return this.require(sourceId).search(query);
  }

  getAsset(sourceId: string, assetId: string): Promise<ArcRemoteAsset | null> {
    return this.require(sourceId).getAsset(assetId);
  }

  getDownloadManifest(sourceId: string, assetId: string): Promise<ArcAssetDownloadManifest> {
    return this.require(sourceId).getDownloadManifest(assetId);
  }
}

export const createDefaultAssetSourceRegistry = (appVersion: string): AssetSourceRegistry =>
  new AssetSourceRegistry([new PolyHavenAssetSource({ userAgent: `ARC-Editor/${appVersion || 'dev'}` })]);
