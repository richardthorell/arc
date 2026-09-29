import type {
  ArcAssetImportRecipe,
  ArcAssetImportRequest,
  ArcImportedAssetProvenance,
  ArcRemoteAsset,
} from './assetSourceTypes';

const normalizeLogicalPaths = (paths: readonly string[]): string[] =>
  [...new Set(paths.map((path) => path.replaceAll('\\', '/').replace(/^\.\//, '').trim()).filter(Boolean))].sort(
    (left, right) => left.localeCompare(right),
  );

const normalizeOptions = (
  options: Record<string, string | number | boolean | null> | undefined,
): Record<string, string | number | boolean | null> | undefined => {
  if (!options) return undefined;
  const entries = Object.entries(options).sort(([left], [right]) => left.localeCompare(right));
  return entries.length > 0 ? Object.fromEntries(entries) : undefined;
};

export const createAssetImportRecipe = (
  request: Pick<ArcAssetImportRequest, 'logicalPaths'>,
  options?: Record<string, string | number | boolean | null>,
): ArcAssetImportRecipe => ({
  version: 1,
  logicalPaths: normalizeLogicalPaths(request.logicalPaths),
  options: normalizeOptions(options),
});

export type CreateAssetProvenanceOptions = {
  importedAt: string;
  sourceUrl?: string;
  sourceRevision?: string;
  sourceHash?: string;
  recipe: ArcAssetImportRecipe;
};

export const createImportedAssetProvenance = (
  asset: Pick<ArcRemoteAsset, 'sourceId' | 'id' | 'license'>,
  options: CreateAssetProvenanceOptions,
): ArcImportedAssetProvenance => ({
  sourceId: asset.sourceId,
  sourceAssetId: asset.id,
  importedAt: options.importedAt,
  license: asset.license,
  sourceUrl: options.sourceUrl,
  sourceRevision: options.sourceRevision,
  sourceHash: options.sourceHash,
  recipe: {
    version: options.recipe.version,
    logicalPaths: [...options.recipe.logicalPaths],
    options: options.recipe.options ? { ...options.recipe.options } : undefined,
  },
});

export const createReimportRequest = (provenance: ArcImportedAssetProvenance): ArcAssetImportRequest | null => {
  if (!provenance.recipe) return null;
  return {
    sourceId: provenance.sourceId,
    assetId: provenance.sourceAssetId,
    logicalPaths: [...provenance.recipe.logicalPaths],
    destinationScope: 'project',
  };
};

export type ArcAssetProvenanceMetadata = {
  provider: string;
  providerAssetId: string;
  importedAt: string;
  licenseAtImport: string;
  sourceUrl?: string;
  sourceRevision?: string;
  sourceHash?: string;
  recipeVersion?: number;
  selectedFiles: string[];
  importOptions: ReadonlyArray<{ key: string; value: string | number | boolean | null }>;
};

export const createAssetProvenanceMetadata = (provenance: ArcImportedAssetProvenance): ArcAssetProvenanceMetadata => ({
  provider: provenance.sourceId,
  providerAssetId: provenance.sourceAssetId,
  importedAt: provenance.importedAt,
  licenseAtImport: provenance.license,
  sourceUrl: provenance.sourceUrl,
  sourceRevision: provenance.sourceRevision,
  sourceHash: provenance.sourceHash,
  recipeVersion: provenance.recipe?.version,
  selectedFiles: provenance.recipe ? [...provenance.recipe.logicalPaths] : [],
  importOptions: Object.entries(provenance.recipe?.options ?? {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => ({ key, value })),
});
