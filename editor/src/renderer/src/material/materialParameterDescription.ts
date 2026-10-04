import type { MaterialParameterDescriptor } from './materialParameterGroups';

export type MaterialParameterDescriptionDescriptor = MaterialParameterDescriptor & {
  description?: string;
};

/** Normalizes optional parameter help text for consistent authoring UI. */
export const normalizeMaterialParameterDescription = (description: string | undefined): string | undefined => {
  const normalized = description?.trim();
  return normalized || undefined;
};

/**
 * Updates parameter help text without changing stable identity, grouping, or order.
 * Blank descriptions intentionally clear authored help text.
 */
export function setMaterialParameterDescription<T extends MaterialParameterDescriptionDescriptor>(
  parameters: readonly T[],
  parameterId: string,
  description: string | undefined,
): T[] {
  const normalized = normalizeMaterialParameterDescription(description);
  return parameters.map((parameter) =>
    parameter.id === parameterId ? { ...parameter, description: normalized } : parameter,
  );
}
