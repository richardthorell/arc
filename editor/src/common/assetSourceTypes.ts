export type ArcAssetScope = 'builtin' | 'project' | 'user' | 'organization';

export type ArcRemoteAssetKind = 'hdri' | 'texture' | 'model' | 'material' | 'audio' | 'animation' | 'other';

export type ArcAssetSourceDescriptor = {
  id: string;
  displayName: string;
  homepage: string;
  attribution?: string;
  licenseSummary?: string;
  capabilities: {
    search: boolean;
    downloadManifest: boolean;
  };
};

export type ArcAssetSourceQuery = {
  text?: string;
  kinds?: ArcRemoteAssetKind[];
  limit?: number;
};

export type ArcRemoteAsset = {
  id: string;
  sourceId: string;
  name: string;
  description: string;
  kind: ArcRemoteAssetKind;
  category: string;
  tags: string[];
  thumbnailUrl?: string;
  license: string;
  attribution?: string;
  publishedAt?: string;
  metadata: Record<string, unknown>;
};

export type ArcAssetSearchResult = {
  source: ArcAssetSourceDescriptor;
  assets: ArcRemoteAsset[];
  total: number;
};

export type ArcAssetDownloadFile = {
  logicalPath: string;
  url: string;
  sizeBytes?: number;
  checksum?: {
    algorithm: 'md5' | 'sha256';
    value: string;
  };
};

export type ArcAssetDownloadManifest = {
  sourceId: string;
  assetId: string;
  files: ArcAssetDownloadFile[];
};

export type ArcImportedAssetProvenance = {
  sourceId: string;
  sourceAssetId: string;
  importedAt: string;
  license: string;
  sourceUrl?: string;
  sourceRevision?: string;
};

export type ArcAssetImportRequest = {
  sourceId: string;
  assetId: string;
  logicalPaths: string[];
  destinationScope: 'project';
};

export type ArcAssetImportProgress = {
  phase: 'resolving' | 'downloading' | 'verifying' | 'copying' | 'complete';
  completedFiles: number;
  totalFiles: number;
  completedBytes: number;
  totalBytes?: number;
  currentFile?: string;
};

export type ArcAssetImportResult = {
  succeeded: boolean;
  destinationRoot: string;
  importedFiles: string[];
  cacheHits: number;
  downloadedFiles: number;
  provenance: ArcImportedAssetProvenance;
};

export interface ArcAssetSourceAdapter {
  readonly descriptor: ArcAssetSourceDescriptor;
  search(query?: ArcAssetSourceQuery): Promise<ArcAssetSearchResult>;
  getAsset(assetId: string): Promise<ArcRemoteAsset | null>;
  getDownloadManifest(assetId: string): Promise<ArcAssetDownloadManifest>;
}
