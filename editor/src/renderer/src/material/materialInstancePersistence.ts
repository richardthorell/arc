export const MATERIAL_INSTANCE_ASSET_VERSION = 1 as const;

export type PersistedMaterialInstanceOverride = {
  parameterId: string;
  value: unknown;
};

export type PersistedMaterialInstanceAsset = {
  version: typeof MATERIAL_INSTANCE_ASSET_VERSION;
  parentMaterialId: string;
  overrides: PersistedMaterialInstanceOverride[];
};

export type MaterialInstanceAsset = {
  parentMaterialId: string;
  overrides: PersistedMaterialInstanceOverride[];
};

/** Serialize only authored instance state. Parent defaults remain owned by the parent material. */
export function serializeMaterialInstanceAsset(asset: MaterialInstanceAsset): string {
  const parentMaterialId = asset.parentMaterialId.trim();
  if (!parentMaterialId) throw new Error('Material instance requires a parent material id.');

  const seen = new Set<string>();
  const overrides = asset.overrides.map((override) => {
    const parameterId = override.parameterId.trim();
    if (!parameterId) throw new Error('Material instance override requires a parameter id.');
    if (seen.has(parameterId)) throw new Error(`Duplicate material instance override: ${parameterId}`);
    seen.add(parameterId);
    return { parameterId, value: override.value };
  });

  return JSON.stringify({ version: MATERIAL_INSTANCE_ASSET_VERSION, parentMaterialId, overrides });
}

/** Load the current authored instance contract without guessing migrations for unsupported versions. */
export function deserializeMaterialInstanceAsset(serialized: string): MaterialInstanceAsset | null {
  let candidate: unknown;
  try {
    candidate = JSON.parse(serialized);
  } catch {
    return null;
  }

  if (!candidate || typeof candidate !== 'object') return null;
  const record = candidate as Record<string, unknown>;
  if (record.version !== MATERIAL_INSTANCE_ASSET_VERSION || typeof record.parentMaterialId !== 'string') return null;
  const parentMaterialId = record.parentMaterialId.trim();
  if (!parentMaterialId || !Array.isArray(record.overrides)) return null;

  const seen = new Set<string>();
  const overrides: PersistedMaterialInstanceOverride[] = [];
  for (const candidateOverride of record.overrides) {
    if (!candidateOverride || typeof candidateOverride !== 'object') return null;
    const override = candidateOverride as Record<string, unknown>;
    if (typeof override.parameterId !== 'string') return null;
    const parameterId = override.parameterId.trim();
    if (!parameterId || seen.has(parameterId) || !Object.prototype.hasOwnProperty.call(override, 'value')) return null;
    seen.add(parameterId);
    overrides.push({ parameterId, value: override.value });
  }

  return { parentMaterialId, overrides };
}
