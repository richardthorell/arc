export type MaterialInstanceParentParameter<T = unknown> = {
  id: string;
  name: string;
  value: T;
};

export type MaterialInstanceOverride<T = unknown> = {
  parameterId: string;
  value: T;
};

export type MaterialInstanceParameterState<T = unknown> = {
  id: string;
  name: string;
  inheritedValue: T;
  value: T;
  isOverridden: boolean;
  canResetToParent: boolean;
};

export type MaterialInstancePresentation<T = unknown> = {
  parameters: MaterialInstanceParameterState<T>[];
  orphanOverrideIds: string[];
  duplicateOverrideIds: string[];
};

/**
 * Build editor presentation state for a material instance without resolving or compiling the material.
 *
 * Parent parameter order remains authoritative. Overrides only replace displayed values; parent identity and
 * inherited values are retained so Inspector controls can show override state and reset-to-parent affordances.
 * Invalid/orphaned override metadata is reported instead of being silently presented as a valid parameter.
 */
export const materialInstancePresentation = <T>(
  parentParameters: readonly MaterialInstanceParentParameter<T>[],
  overrides: readonly MaterialInstanceOverride<T>[],
): MaterialInstancePresentation<T> => {
  const parentIds = new Set(parentParameters.map((parameter) => parameter.id));
  const overrideValues = new Map<string, T>();
  const duplicateOverrideIds = new Set<string>();
  const orphanOverrideIds = new Set<string>();

  for (const override of overrides) {
    if (!parentIds.has(override.parameterId)) orphanOverrideIds.add(override.parameterId);
    if (overrideValues.has(override.parameterId)) duplicateOverrideIds.add(override.parameterId);
    else overrideValues.set(override.parameterId, override.value);
  }

  return {
    parameters: parentParameters.map((parameter) => {
      const isOverridden = overrideValues.has(parameter.id);
      return {
        id: parameter.id,
        name: parameter.name,
        inheritedValue: parameter.value,
        value: isOverridden ? (overrideValues.get(parameter.id) as T) : parameter.value,
        isOverridden,
        canResetToParent: isOverridden,
      };
    }),
    orphanOverrideIds: [...orphanOverrideIds].sort(),
    duplicateOverrideIds: [...duplicateOverrideIds].sort(),
  };
};
