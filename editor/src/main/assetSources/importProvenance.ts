import type {
  ArcAssetDownloadManifest,
  ArcAssetImportRecipe,
  ArcImportedAssetProvenance,
  ArcRemoteAsset,
} from '../../common/assetSourceTypes';

export type BuildImportProvenanceInput = {
  asset: ArcRemoteAsset;
  manifest: ArcAssetDownloadManifest;
  logicalPaths: string[];
  importedAt: string;
  options?: ArcAssetImportRecipe['options'];
};

const normalizedLogicalPaths = (paths: string[]): string[] =>
  [...new Set(paths.map((path) => path.trim()).filter(Boolean))].sort((left, right) => left.localeCompare(right));

const sourceRevision = (asset: ArcRemoteAsset): string | undefined => {
  const revision = asset.metadata.filesHash;
  return typeof revision === 'string' && revision.trim() ? revision.trim() : undefined;
};

const sourceUrl = (asset: ArcRemoteAsset): string | undefined => {
  const url = asset.metadata.sourceUrl;
  return typeof url === 'string' && url.trim() ? url.trim() : undefined;
};

export const buildImportProvenance = ({
  asset,
  manifest,
  logicalPaths,
  importedAt,
  options,
}: BuildImportProvenanceInput): ArcImportedAssetProvenance => {
  if (asset.sourceId !== manifest.sourceId || asset.id !== manifest.assetId) {
    throw new Error('Import provenance asset and download manifest identities must match');
  }

  const selectedPaths = normalizedLogicalPaths(logicalPaths);
  if (selectedPaths.length === 0) throw new Error('Import provenance requires at least one selected source file');

  const availablePaths = new Set(manifest.files.map((file) => file.logicalPath));
  const missingPath = selectedPaths.find((path) => !availablePaths.has(path));
  if (missingPath) throw new Error(`Import provenance references unknown source file '${missingPath}'`);

  const revision = sourceRevision(asset);
  const provenance: ArcImportedAssetProvenance = {
    sourceId: asset.sourceId,
    sourceAssetId: asset.id,
    importedAt,
    license: asset.license,
    recipe: {
      version: 1,
      logicalPaths: selectedPaths,
      ...(options ? { options: { ...options } } : {}),
    },
  };

  const url = sourceUrl(asset);
  if (url) provenance.sourceUrl = url;
  if (revision) provenance.sourceRevision = revision;

  return provenance;
};
