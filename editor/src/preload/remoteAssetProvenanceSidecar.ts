import type { ArcAssetImportRecipe, ArcImportedAssetProvenance } from '../common/assetSourceTypes';
import type { ArcRemoteImportProvenanceSidecar } from './remoteAssetProvenance';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((entry) => typeof entry === 'string');

const isRecipeOption = (value: unknown): value is string | number | boolean | null =>
  value === null || ['string', 'number', 'boolean'].includes(typeof value);

const parseRecipe = (value: unknown): ArcAssetImportRecipe | undefined => {
  if (value === undefined) return undefined;
  if (!isRecord(value) || value.version !== 1 || !isStringArray(value.logicalPaths)) return undefined;

  let options: ArcAssetImportRecipe['options'];
  if (value.options !== undefined) {
    if (!isRecord(value.options) || !Object.values(value.options).every(isRecipeOption)) return undefined;
    options = { ...value.options } as NonNullable<ArcAssetImportRecipe['options']>;
  }

  return {
    version: 1,
    logicalPaths: [...value.logicalPaths],
    options,
  };
};

const parseProvenance = (value: unknown): ArcImportedAssetProvenance | null => {
  if (!isRecord(value)) return null;
  if (
    typeof value.sourceId !== 'string' ||
    typeof value.sourceAssetId !== 'string' ||
    typeof value.importedAt !== 'string' ||
    typeof value.license !== 'string'
  ) {
    return null;
  }

  for (const field of ['sourceUrl', 'sourceRevision', 'sourceHash'] as const) {
    if (value[field] !== undefined && typeof value[field] !== 'string') return null;
  }

  const recipe = parseRecipe(value.recipe);
  if (value.recipe !== undefined && !recipe) return null;

  return {
    sourceId: value.sourceId,
    sourceAssetId: value.sourceAssetId,
    importedAt: value.importedAt,
    license: value.license,
    sourceUrl: value.sourceUrl as string | undefined,
    sourceRevision: value.sourceRevision as string | undefined,
    sourceHash: value.sourceHash as string | undefined,
    recipe,
  };
};

export const parseRemoteImportProvenanceSidecar = (serialized: string): ArcRemoteImportProvenanceSidecar | null => {
  let value: unknown;
  try {
    value = JSON.parse(serialized) as unknown;
  } catch {
    return null;
  }

  if (!isRecord(value) || value.version !== 1) return null;
  if (!isStringArray(value.importedFiles) || !isStringArray(value.importedAssetIds)) return null;

  const provenance = parseProvenance(value.provenance);
  if (!provenance) return null;

  return {
    version: 1,
    provenance,
    importedFiles: [...value.importedFiles],
    importedAssetIds: [...value.importedAssetIds],
  };
};
