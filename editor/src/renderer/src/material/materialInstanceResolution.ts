export type MaterialInstanceParameterValue = boolean | number | string | readonly number[];

export interface MaterialInstanceParameter {
  id: string;
  name: string;
  value: MaterialInstanceParameterValue;
}

export interface MaterialInstanceOverride {
  parameterId: string;
  value: MaterialInstanceParameterValue;
}

export interface MaterialInstanceAsset {
  id: string;
  parentMaterialId: string;
  overrides: readonly MaterialInstanceOverride[];
}

export interface ResolvedMaterialInstanceParameter extends MaterialInstanceParameter {
  inherited: boolean;
}

export interface ResolvedMaterialInstance {
  instanceId: string;
  parentMaterialId: string;
  parameters: readonly ResolvedMaterialInstanceParameter[];
  staleOverrideIds: readonly string[];
}

function cloneValue(value: MaterialInstanceParameterValue): MaterialInstanceParameterValue {
  return Array.isArray(value) ? [...value] : value;
}

/**
 * Resolves instance values against the parent's reflected parameter set without
 * mutating either input. Parameter identity is the stable reflected id rather
 * than the display name, so parent renames do not invalidate overrides.
 *
 * The parent's order remains authoritative for deterministic preview/runtime
 * presentation. Unknown overrides are reported instead of silently becoming
 * parameters, allowing callers to surface stale data and offer reset/cleanup.
 */
export function resolveMaterialInstance(
  instance: MaterialInstanceAsset,
  parentParameters: readonly MaterialInstanceParameter[],
): ResolvedMaterialInstance {
  const parentIds = new Set(parentParameters.map((parameter) => parameter.id));
  const overrides = new Map<string, MaterialInstanceParameterValue>();
  const staleOverrideIds = new Set<string>();

  for (const override of instance.overrides) {
    if (!parentIds.has(override.parameterId)) {
      staleOverrideIds.add(override.parameterId);
      continue;
    }

    // Persisted data should contain one override per stable parameter id. If an
    // older asset contains duplicates, the last authored value wins.
    overrides.set(override.parameterId, cloneValue(override.value));
  }

  return {
    instanceId: instance.id,
    parentMaterialId: instance.parentMaterialId,
    parameters: parentParameters.map((parameter) => {
      const override = overrides.get(parameter.id);
      return {
        ...parameter,
        value: cloneValue(override === undefined ? parameter.value : override),
        inherited: override === undefined,
      };
    }),
    staleOverrideIds: [...staleOverrideIds].sort(),
  };
}

/** Return a new instance asset with one override reset to its parent value. */
export function resetMaterialInstanceOverride(
  instance: MaterialInstanceAsset,
  parameterId: string,
): MaterialInstanceAsset {
  return {
    ...instance,
    overrides: instance.overrides.filter((override) => override.parameterId !== parameterId),
  };
}
