import type { MaterialAssetJson } from './materialGraphTypes';
import type { MaterialParameterDescriptor } from './materialParameterGroups';

export const MATERIAL_PARAMETER_METADATA_VERSION = 1;

export type MaterialParameterAuthoringMetadata = {
  version: typeof MATERIAL_PARAMETER_METADATA_VERSION;
  parameters: Record<string, { group?: string; order?: number }>;
};

const normalizedEntry = (value: unknown): { group?: string; order?: number } | undefined => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const source = value as Record<string, unknown>;
  const group = typeof source.group === 'string' ? source.group.trim() : '';
  const order = typeof source.order === 'number' && Number.isInteger(source.order) && source.order >= 0 ? source.order : undefined;
  if (!group && order === undefined) return undefined;
  return { ...(group ? { group } : {}), ...(order !== undefined ? { order } : {}) };
};

export const materialParameterMetadataFromAsset = (asset: MaterialAssetJson): MaterialParameterAuthoringMetadata => {
  const raw = asset.parameterMetadata;
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return { version: MATERIAL_PARAMETER_METADATA_VERSION, parameters: {} };
  }

  const source = raw as Record<string, unknown>;
  if (source.version !== MATERIAL_PARAMETER_METADATA_VERSION || !source.parameters || typeof source.parameters !== 'object' || Array.isArray(source.parameters)) {
    return { version: MATERIAL_PARAMETER_METADATA_VERSION, parameters: {} };
  }

  const parameters: MaterialParameterAuthoringMetadata['parameters'] = {};
  for (const [id, value] of Object.entries(source.parameters as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right))) {
    const entry = normalizedEntry(value);
    if (id && entry) parameters[id] = entry;
  }
  return { version: MATERIAL_PARAMETER_METADATA_VERSION, parameters };
};

export const withMaterialParameterMetadata = (
  asset: MaterialAssetJson,
  descriptors: readonly MaterialParameterDescriptor[],
): MaterialAssetJson => {
  const parameters: MaterialParameterAuthoringMetadata['parameters'] = {};
  for (const descriptor of [...descriptors].sort((left, right) => left.id.localeCompare(right.id))) {
    const entry = normalizedEntry({ group: descriptor.group, order: descriptor.order });
    if (descriptor.id && entry) parameters[descriptor.id] = entry;
  }

  if (Object.keys(parameters).length === 0) {
    const { parameterMetadata: _parameterMetadata, ...rest } = asset;
    return rest;
  }
  return {
    ...asset,
    parameterMetadata: { version: MATERIAL_PARAMETER_METADATA_VERSION, parameters },
  };
};

export const applyMaterialParameterMetadata = <T extends MaterialParameterDescriptor>(
  descriptors: readonly T[],
  metadata: MaterialParameterAuthoringMetadata,
): T[] =>
  descriptors.map((descriptor) => {
    const entry = metadata.parameters[descriptor.id];
    return entry ? { ...descriptor, ...entry } : { ...descriptor };
  });
