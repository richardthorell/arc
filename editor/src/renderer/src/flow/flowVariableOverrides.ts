import type { FlowValueType, FlowVariableDefinition } from './flowGraphTypes';

export type FlowVariableOverride = {
  variableId: string;
  value: unknown;
};

export type FlowExposedVariable = {
  id: string;
  name: string;
  type: FlowValueType;
  defaultValue: unknown;
  value: unknown;
  overridden: boolean;
};

const exposedVariables = (variables: readonly FlowVariableDefinition[]) =>
  variables.filter((variable) => variable.exposed);

export function resolveFlowVariableOverrides(
  variables: readonly FlowVariableDefinition[],
  overrides: readonly FlowVariableOverride[],
): FlowExposedVariable[] {
  const overridesById = new Map(overrides.map((override) => [override.variableId, override.value]));

  return exposedVariables(variables).map((variable) => ({
    id: variable.id,
    name: variable.name,
    type: variable.type,
    defaultValue: variable.defaultValue,
    value: overridesById.has(variable.id) ? overridesById.get(variable.id) : variable.defaultValue,
    overridden: overridesById.has(variable.id),
  }));
}

export function setFlowVariableOverride(
  overrides: readonly FlowVariableOverride[],
  variableId: string,
  value: unknown,
): FlowVariableOverride[] {
  const index = overrides.findIndex((override) => override.variableId === variableId);
  if (index < 0) {
    return [...overrides, { variableId, value }];
  }

  const next = [...overrides];
  next[index] = { variableId, value };
  return next;
}

export function resetFlowVariableOverride(
  overrides: readonly FlowVariableOverride[],
  variableId: string,
): FlowVariableOverride[] {
  return overrides.filter((override) => override.variableId !== variableId);
}

export function reconcileFlowVariableOverrides(
  variables: readonly FlowVariableDefinition[],
  overrides: readonly FlowVariableOverride[],
): FlowVariableOverride[] {
  const exposedById = new Map(exposedVariables(variables).map((variable) => [variable.id, variable]));
  const seen = new Set<string>();

  return overrides.filter((override) => {
    if (seen.has(override.variableId) || !exposedById.has(override.variableId)) {
      return false;
    }
    seen.add(override.variableId);
    return true;
  });
}
