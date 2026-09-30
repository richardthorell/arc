export type AssetSourceProvenance = {
  provider: string;
  providerAssetId: string;
  sourceUrl: string;
  sourceRevision?: string;
  sourceHash?: string;
  licenseAtImport?: string;
};

export type AssetImportRecipe = {
  importer: string;
  variant: string;
  options: Readonly<Record<string, string | number | boolean>>;
};

export type AssetProvenanceMetadata = {
  version: 1;
  source: AssetSourceProvenance;
  recipe: AssetImportRecipe;
};

export type AssetReimportRequest = {
  assetId: string;
  source: AssetSourceProvenance;
  recipe: AssetImportRecipe;
};

function requireText(value: string, field: string): string {
  const normalized = value.trim();
  if (!normalized) {
    throw new Error(`Asset provenance ${field} must not be empty`);
  }
  return normalized;
}

function normalizeOptionalText(value: string | undefined): string | undefined {
  const normalized = value?.trim();
  return normalized ? normalized : undefined;
}

function sortedOptions(
  options: Readonly<Record<string, string | number | boolean>>,
): Readonly<Record<string, string | number | boolean>> {
  return Object.fromEntries(Object.entries(options).sort(([left], [right]) => left.localeCompare(right)));
}

/**
 * Captures the source identity and exact import recipe independently from the
 * ARC asset ID. Re-import can therefore reproduce the original import without
 * consulting current UI defaults or changing stable asset identity.
 */
export function createAssetProvenanceMetadata(
  source: AssetSourceProvenance,
  recipe: AssetImportRecipe,
): AssetProvenanceMetadata {
  return {
    version: 1,
    source: {
      provider: requireText(source.provider, 'provider'),
      providerAssetId: requireText(source.providerAssetId, 'provider asset ID'),
      sourceUrl: requireText(source.sourceUrl, 'source URL'),
      sourceRevision: normalizeOptionalText(source.sourceRevision),
      sourceHash: normalizeOptionalText(source.sourceHash),
      licenseAtImport: normalizeOptionalText(source.licenseAtImport),
    },
    recipe: {
      importer: requireText(recipe.importer, 'importer'),
      variant: requireText(recipe.variant, 'variant'),
      options: sortedOptions(recipe.options),
    },
  };
}

export function createAssetReimportRequest(assetId: string, metadata: AssetProvenanceMetadata): AssetReimportRequest {
  return {
    assetId: requireText(assetId, 'asset ID'),
    source: { ...metadata.source },
    recipe: {
      ...metadata.recipe,
      options: { ...metadata.recipe.options },
    },
  };
}
