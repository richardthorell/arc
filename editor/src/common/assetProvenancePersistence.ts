import type { ArcAssetImportRecipe, ArcImportedAssetProvenance } from './assetSourceTypes';

export type ArcAssetProvenanceSidecar = {
  version: 1;
  provenance: ArcImportedAssetProvenance;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isRecipe = (value: unknown): value is ArcAssetImportRecipe => {
  if (!isRecord(value) || value.version !== 1 || !Array.isArray(value.logicalPaths)) return false;
  if (!value.logicalPaths.every((path) => typeof path === 'string')) return false;
  if (value.options === undefined) return true;
  if (!isRecord(value.options)) return false;
  return Object.values(value.options).every(
    (option) => option === null || ['string', 'number', 'boolean'].includes(typeof option),
  );
};

const isOptionalString = (value: unknown): value is string | undefined =>
  value === undefined || typeof value === 'string';

const isProvenance = (value: unknown): value is ArcImportedAssetProvenance => {
  if (!isRecord(value)) return false;
  if (
    typeof value.sourceId !== 'string' ||
    typeof value.sourceAssetId !== 'string' ||
    typeof value.importedAt !== 'string' ||
    typeof value.license !== 'string'
  ) {
    return false;
  }
  if (
    !isOptionalString(value.sourceUrl) ||
    !isOptionalString(value.sourceRevision) ||
    !isOptionalString(value.sourceHash)
  ) {
    return false;
  }
  return value.recipe === undefined || isRecipe(value.recipe);
};

export const serializeAssetProvenanceSidecar = (provenance: ArcImportedAssetProvenance): string =>
  `${JSON.stringify({ version: 1, provenance } satisfies ArcAssetProvenanceSidecar, null, 2)}\n`;

export const parseAssetProvenanceSidecar = (contents: string): ArcImportedAssetProvenance | null => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(contents);
  } catch {
    return null;
  }

  if (!isRecord(parsed) || parsed.version !== 1 || !isProvenance(parsed.provenance)) return null;

  return {
    ...parsed.provenance,
    recipe: parsed.provenance.recipe
      ? {
          ...parsed.provenance.recipe,
          logicalPaths: [...parsed.provenance.recipe.logicalPaths],
          options: parsed.provenance.recipe.options ? { ...parsed.provenance.recipe.options } : undefined,
        }
      : undefined,
  };
};
