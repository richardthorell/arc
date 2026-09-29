export interface MaterialParameterDescriptor {
  id: string;
  name: string;
  group?: string;
  order?: number;
}

export interface MaterialParameterGroup<T extends MaterialParameterDescriptor = MaterialParameterDescriptor> {
  id: string;
  label: string;
  parameters: T[];
}

export const DEFAULT_MATERIAL_PARAMETER_GROUP_ID = 'general';
export const DEFAULT_MATERIAL_PARAMETER_GROUP_LABEL = 'General';

function normalizeGroup(group: string | undefined): { id: string; label: string } {
  const label = group?.trim();
  if (!label) {
    return {
      id: DEFAULT_MATERIAL_PARAMETER_GROUP_ID,
      label: DEFAULT_MATERIAL_PARAMETER_GROUP_LABEL,
    };
  }

  return { id: label.toLocaleLowerCase(), label };
}

function sameGroup(left: string | undefined, right: string | undefined): boolean {
  return normalizeGroup(left).id === normalizeGroup(right).id;
}

/**
 * Groups material parameters for authoring without making presentation metadata
 * part of parameter identity. Ordering is deterministic so save/load and graph
 * discovery cannot reshuffle the UI between sessions.
 */
export function groupMaterialParameters<T extends MaterialParameterDescriptor>(
  parameters: readonly T[],
): MaterialParameterGroup<T>[] {
  const groups = new Map<string, MaterialParameterGroup<T>>();

  for (const parameter of parameters) {
    const normalized = normalizeGroup(parameter.group);
    const existing = groups.get(normalized.id);
    if (existing) {
      existing.parameters.push(parameter);
    } else {
      groups.set(normalized.id, {
        id: normalized.id,
        label: normalized.label,
        parameters: [parameter],
      });
    }
  }

  const compareParameters = (left: T, right: T): number => {
    const order = (left.order ?? 0) - (right.order ?? 0);
    if (order !== 0) return order;
    const name = left.name.localeCompare(right.name);
    if (name !== 0) return name;
    return left.id.localeCompare(right.id);
  };

  for (const group of groups.values()) {
    group.parameters.sort(compareParameters);
  }

  return [...groups.values()].sort((left, right) => {
    if (left.id === DEFAULT_MATERIAL_PARAMETER_GROUP_ID) return -1;
    if (right.id === DEFAULT_MATERIAL_PARAMETER_GROUP_ID) return 1;
    return left.label.localeCompare(right.label) || left.id.localeCompare(right.id);
  });
}

/**
 * Assigns one parameter to an authoring group without changing its stable id.
 * Blank group names intentionally mean the shared General group.
 */
export function assignMaterialParameterGroup<T extends MaterialParameterDescriptor>(
  parameters: readonly T[],
  parameterId: string,
  group: string | undefined,
): T[] {
  const normalized = group?.trim() || undefined;
  return parameters.map((parameter) =>
    parameter.id === parameterId ? { ...parameter, group: normalized, order: undefined } : parameter,
  );
}

/**
 * Reorders a parameter within its current group. Orders are compacted to stable
 * zero-based values so repeated drag/reorder operations cannot accumulate gaps.
 */
export function reorderMaterialParameter<T extends MaterialParameterDescriptor>(
  parameters: readonly T[],
  parameterId: string,
  targetIndex: number,
): T[] {
  const source = parameters.find((parameter) => parameter.id === parameterId);
  if (!source) return [...parameters];

  const group = groupMaterialParameters(parameters).find((candidate) =>
    candidate.parameters.some((parameter) => parameter.id === parameterId),
  );
  if (!group) return [...parameters];

  const orderedIds = group.parameters.map((parameter) => parameter.id);
  const sourceIndex = orderedIds.indexOf(parameterId);
  orderedIds.splice(sourceIndex, 1);
  const clampedIndex = Math.max(0, Math.min(targetIndex, orderedIds.length));
  orderedIds.splice(clampedIndex, 0, parameterId);

  const orderById = new Map(orderedIds.map((id, index) => [id, index]));
  return parameters.map((parameter) => {
    if (!sameGroup(parameter.group, source.group)) return parameter;
    return { ...parameter, order: orderById.get(parameter.id) };
  });
}
