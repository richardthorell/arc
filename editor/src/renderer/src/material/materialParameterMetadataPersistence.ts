import { normalizeMaterialParameterMetadata, type MaterialParameterDescriptor } from './materialParameterGroups';

export const MATERIAL_PARAMETER_METADATA_VERSION = 1;

export interface MaterialParameterAuthoringMetadata extends MaterialParameterDescriptor {
  description?: string;
}

export interface PersistedMaterialParameterMetadata {
  version: typeof MATERIAL_PARAMETER_METADATA_VERSION;
  parameters: Array<{
    id: string;
    group?: string;
    order: number;
    description?: string;
  }>;
}

function normalizeDescription(description: string | undefined): string | undefined {
  const normalized = description?.trim();
  return normalized || undefined;
}

/**
 * Builds the versioned presentation-metadata payload stored alongside compiler-owned
 * material parameter semantics. Stable parameter IDs are the only identity carried
 * across save/load; names and types remain authoritative in the material graph.
 */
export function serializeMaterialParameterMetadata(
  parameters: readonly MaterialParameterAuthoringMetadata[],
): PersistedMaterialParameterMetadata {
  const seen = new Set<string>();
  for (const parameter of parameters) {
    if (!parameter.id || seen.has(parameter.id)) {
      throw new Error(`Material parameter metadata requires unique non-empty ids: ${parameter.id || '<empty>'}`);
    }
    seen.add(parameter.id);
  }

  const normalized = normalizeMaterialParameterMetadata(parameters);
  return {
    version: MATERIAL_PARAMETER_METADATA_VERSION,
    parameters: normalized
      .map((parameter) => ({
        id: parameter.id,
        group: parameter.group,
        order: parameter.order ?? 0,
        description: normalizeDescription(parameter.description),
      }))
      .sort((left, right) => left.id.localeCompare(right.id)),
  };
}

/**
 * Validates persisted authoring metadata before it is merged with graph-discovered
 * parameters. Unknown/stale IDs are retained in the payload so callers can report
 * them explicitly instead of silently rebinding metadata by display name.
 */
export function parseMaterialParameterMetadata(value: unknown): PersistedMaterialParameterMetadata | null {
  if (!value || typeof value !== 'object') return null;
  const candidate = value as { version?: unknown; parameters?: unknown };
  if (candidate.version !== MATERIAL_PARAMETER_METADATA_VERSION || !Array.isArray(candidate.parameters)) {
    return null;
  }

  const ids = new Set<string>();
  const parameters: PersistedMaterialParameterMetadata['parameters'] = [];
  for (const raw of candidate.parameters) {
    if (!raw || typeof raw !== 'object') return null;
    const entry = raw as { id?: unknown; group?: unknown; order?: unknown; description?: unknown };
    if (typeof entry.id !== 'string' || !entry.id || ids.has(entry.id)) return null;
    if (entry.group !== undefined && typeof entry.group !== 'string') return null;
    if (typeof entry.order !== 'number' || !Number.isSafeInteger(entry.order) || entry.order < 0) return null;
    if (entry.description !== undefined && typeof entry.description !== 'string') return null;

    ids.add(entry.id);
    parameters.push({
      id: entry.id,
      group: entry.group?.trim() || undefined,
      order: entry.order,
      description: normalizeDescription(entry.description),
    });
  }

  parameters.sort((left, right) => left.id.localeCompare(right.id));
  return { version: MATERIAL_PARAMETER_METADATA_VERSION, parameters };
}
