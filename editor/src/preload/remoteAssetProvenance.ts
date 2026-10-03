import { createHash } from 'node:crypto';

import { createAssetImportRecipe, createImportedAssetProvenance } from '../common/assetProvenance';
import type {
  ArcAssetDownloadFile,
  ArcAssetImportRequest,
  ArcImportedAssetProvenance,
  ArcRemoteAsset,
} from '../common/assetSourceTypes';

export type CreateRemoteImportProvenanceOptions = {
  importedAt: string;
  sourceHomepage?: string;
  importOptions?: Record<string, string | number | boolean | null>;
};

const normalizeChecksum = (file: ArcAssetDownloadFile): string =>
  file.checksum
    ? `${file.checksum.algorithm}:${file.checksum.value.toLocaleLowerCase()}`
    : `url:${file.url}|size:${file.sizeBytes ?? 'unknown'}`;

export const createRemoteManifestHash = (files: readonly ArcAssetDownloadFile[]): string => {
  const canonical = files
    .map((file) => `${file.logicalPath.replaceAll('\\', '/')}|${normalizeChecksum(file)}`)
    .sort((left, right) => left.localeCompare(right))
    .join('\n');
  return `sha256:${createHash('sha256').update(canonical).digest('hex')}`;
};

export const createRemoteImportProvenance = (
  asset: ArcRemoteAsset,
  request: Pick<ArcAssetImportRequest, 'logicalPaths'>,
  selectedFiles: readonly ArcAssetDownloadFile[],
  options: CreateRemoteImportProvenanceOptions,
): ArcImportedAssetProvenance => {
  const sourceRevision = typeof asset.metadata.filesHash === 'string' ? asset.metadata.filesHash : undefined;
  const sourceUrl = options.sourceHomepage
    ? `${options.sourceHomepage.replace(/\/$/, '')}/a/${encodeURIComponent(asset.id)}`
    : undefined;

  return createImportedAssetProvenance(asset, {
    importedAt: options.importedAt,
    sourceUrl,
    sourceRevision,
    sourceHash: createRemoteManifestHash(selectedFiles),
    recipe: createAssetImportRecipe(request, options.importOptions),
  });
};
