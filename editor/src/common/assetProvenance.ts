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

export type ArcDeterministicReimportPlan = {
  arcAssetId: string;
  request: ArcAssetImportRequest;
  recipe: ArcAssetImportRecipe;
  expectedSourceRevision?: string;
  expectedSourceHash?: string;
};

export const createDeterministicReimportPlan = (
  arcAssetId: string,
  provenance: ArcImportedAssetProvenance,
): ArcDeterministicReimportPlan | null => {
  const stableAssetId = arcAssetId.trim();
  const request = createReimportRequest(provenance);
  if (!stableAssetId || !request || !provenance.recipe) return null;

  return {
    arcAssetId: stableAssetId,
    request,
    recipe: {
      version: provenance.recipe.version,
      logicalPaths: [...provenance.recipe.logicalPaths],
      options: provenance.recipe.options ? { ...provenance.recipe.options } : undefined,
    },
    expectedSourceRevision: provenance.sourceRevision,
    expectedSourceHash: provenance.sourceHash,
  };
};

export type ArcCompletedReimport = {
  arcAssetId: string;
  provenance: ArcImportedAssetProvenance;
};

export type CompleteDeterministicReimportOptions = {
  importedAt: string;
  sourceUrl?: string;
  sourceRevision?: string;
  sourceHash?: string;
  license?: string;
};

export const completeDeterministicReimport = (
  plan: ArcDeterministicReimportPlan,
  previous: ArcImportedAssetProvenance,
  options: CompleteDeterministicReimportOptions,
): ArcCompletedReimport | null => {
  if (
    plan.request.sourceId !== previous.sourceId ||
    plan.request.assetId !== previous.sourceAssetId ||
    plan.recipe.version !== previous.recipe?.version
  ) {
    return null;
  }

  return {
    arcAssetId: plan.arcAssetId,
    provenance: {
      sourceId: previous.sourceId,
      sourceAssetId: previous.sourceAssetId,
      importedAt: options.importedAt,
      license: options.license ?? previous.license,
      sourceUrl: options.sourceUrl ?? previous.sourceUrl,
      sourceRevision: options.sourceRevision ?? previous.sourceRevision,
      sourceHash: options.sourceHash ?? previous.sourceHash,
      recipe: {
        version: plan.recipe.version,
        logicalPaths: [...plan.recipe.logicalPaths],
        options: plan.recipe.options ? { ...plan.recipe.options } : undefined,
      },
    },
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
