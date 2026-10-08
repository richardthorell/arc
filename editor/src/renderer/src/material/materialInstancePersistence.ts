export const MATERIAL_INSTANCE_ASSET_VERSION = 1 as const;

export type MaterialInstanceAssetReference = {
  guid: string;
  pathHint: string;
};

export type PersistedMaterialInstanceOverride = {
  parameterId: string;
  value: unknown;
};

export type PersistedMaterialInstanceFunctionInputOverride = {
  pinId: string;
  value: unknown;
};

export type PersistedMaterialInstanceFunctionOverride = {
  slotId: string;
  function: MaterialInstanceAssetReference;
  inputOverrides: PersistedMaterialInstanceFunctionInputOverride[];
};

export type MaterialInstanceAsset = {
  version: typeof MATERIAL_INSTANCE_ASSET_VERSION;
  name: string;
  parent: MaterialInstanceAssetReference;
  parameterOverrides: PersistedMaterialInstanceOverride[];
  functionOverrides: PersistedMaterialInstanceFunctionOverride[];
};

const cleanReference = (reference: MaterialInstanceAssetReference): MaterialInstanceAssetReference => {
  const guid = reference.guid.trim();
  const pathHint = reference.pathHint.trim().replaceAll('\\', '/');
  if (!guid || !pathHint) throw new Error('Material instance asset references require a GUID and path hint.');
  return { guid, pathHint };
};

const normalizedOverrideId = (value: string, label: string) => {
  const id = value.trim();
  if (!id) throw new Error(`${label} requires a stable id.`);
  return id;
};

/** Serialize only authored instance state. Parent defaults remain owned by the parent Material. */
export function serializeMaterialInstanceAsset(asset: MaterialInstanceAsset): string {
  const name = asset.name.trim();
  if (!name) throw new Error('Material instance requires a name.');

  const parameterIds = new Set<string>();
  const parameterOverrides = asset.parameterOverrides.map((override) => {
    const parameterId = normalizedOverrideId(override.parameterId, 'Material instance parameter override');
    if (parameterIds.has(parameterId)) throw new Error(`Duplicate material instance override: ${parameterId}`);
    parameterIds.add(parameterId);
    return { parameterId, value: override.value };
  });

  const slotIds = new Set<string>();
  const functionOverrides = asset.functionOverrides.map((override) => {
    const slotId = normalizedOverrideId(override.slotId, 'Material instance Function Slot override');
    if (slotIds.has(slotId)) throw new Error(`Duplicate material instance Function Slot override: ${slotId}`);
    slotIds.add(slotId);
    const pinIds = new Set<string>();
    const inputOverrides = override.inputOverrides.map((input) => {
      const pinId = normalizedOverrideId(input.pinId, 'Material instance Function input override');
      if (pinIds.has(pinId)) throw new Error(`Duplicate Function input override: ${slotId}/${pinId}`);
      pinIds.add(pinId);
      return { pinId, value: input.value };
    });
    return { slotId, function: cleanReference(override.function), inputOverrides };
  });

  return `${JSON.stringify(
    {
      version: MATERIAL_INSTANCE_ASSET_VERSION,
      name,
      parent: cleanReference(asset.parent),
      parameterOverrides,
      functionOverrides,
    },
    null,
    2,
  )}\n`;
}

const parseReference = (value: unknown): MaterialInstanceAssetReference | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  if (typeof record.guid !== 'string' || typeof record.pathHint !== 'string') return null;
  const guid = record.guid.trim();
  const pathHint = record.pathHint.trim().replaceAll('\\', '/');
  return guid && pathHint ? { guid, pathHint } : null;
};

/** Load the native Material Instance v1 authoring contract without guessing migrations. */
export function deserializeMaterialInstanceAsset(serialized: string): MaterialInstanceAsset | null {
  let candidate: unknown;
  try {
    candidate = JSON.parse(serialized);
  } catch {
    return null;
  }
  if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) return null;
  const record = candidate as Record<string, unknown>;
  if (record.version !== MATERIAL_INSTANCE_ASSET_VERSION || typeof record.name !== 'string') return null;
  const name = record.name.trim();
  const parent = parseReference(record.parent);
  if (!name || !parent || !Array.isArray(record.parameterOverrides) || !Array.isArray(record.functionOverrides))
    return null;

  const parameterIds = new Set<string>();
  const parameterOverrides: PersistedMaterialInstanceOverride[] = [];
  for (const candidateOverride of record.parameterOverrides) {
    if (!candidateOverride || typeof candidateOverride !== 'object' || Array.isArray(candidateOverride)) return null;
    const override = candidateOverride as Record<string, unknown>;
    if (typeof override.parameterId !== 'string' || !Object.prototype.hasOwnProperty.call(override, 'value')) return null;
    const parameterId = override.parameterId.trim();
    if (!parameterId || parameterIds.has(parameterId)) return null;
    parameterIds.add(parameterId);
    parameterOverrides.push({ parameterId, value: override.value });
  }

  const slotIds = new Set<string>();
  const functionOverrides: PersistedMaterialInstanceFunctionOverride[] = [];
  for (const candidateOverride of record.functionOverrides) {
    if (!candidateOverride || typeof candidateOverride !== 'object' || Array.isArray(candidateOverride)) return null;
    const override = candidateOverride as Record<string, unknown>;
    const slotId = typeof override.slotId === 'string' ? override.slotId.trim() : '';
    const functionReference = parseReference(override.function);
    if (!slotId || slotIds.has(slotId) || !functionReference || !Array.isArray(override.inputOverrides)) return null;
    slotIds.add(slotId);
    const pinIds = new Set<string>();
    const inputOverrides: PersistedMaterialInstanceFunctionInputOverride[] = [];
    for (const candidateInput of override.inputOverrides) {
      if (!candidateInput || typeof candidateInput !== 'object' || Array.isArray(candidateInput)) return null;
      const input = candidateInput as Record<string, unknown>;
      const pinId = typeof input.pinId === 'string' ? input.pinId.trim() : '';
      if (!pinId || pinIds.has(pinId) || !Object.prototype.hasOwnProperty.call(input, 'value')) return null;
      pinIds.add(pinId);
      inputOverrides.push({ pinId, value: input.value });
    }
    functionOverrides.push({ slotId, function: functionReference, inputOverrides });
  }

  return {
    version: MATERIAL_INSTANCE_ASSET_VERSION,
    name,
    parent,
    parameterOverrides,
    functionOverrides,
  };
}

/** Mirror render::make_shader_parameter_id for authored graph node identity. */
export const materialParameterId = (stableName: string): string => {
  let hash = 14695981039346656037n;
  for (const byte of new TextEncoder().encode(stableName)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 1099511628211n);
  }
  if (hash === 0n) hash = 1n;
  return hash.toString(10);
};

/** Stable parameter id used for a replacement-only Function Slot input. */
export const materialFunctionSlotParameterId = (slotId: string, functionIdentity: string, pinId: string): string =>
  materialParameterId(`slot::${slotId}::${functionIdentity}::${pinId}`);
