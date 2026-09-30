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

export type FlowVariableOverrideDiagnostic = {
  variableId: string;
  reason: 'duplicate' | 'missing-variable' | 'not-exposed' | 'invalid-value';
};

export type FlowVariableOverrideReconciliation = {
  overrides: FlowVariableOverride[];
  diagnostics: FlowVariableOverrideDiagnostic[];
};

const exposedVariables = (variables: readonly FlowVariableDefinition[]) =>
  variables.filter((variable) => variable.exposed);

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const isVector = (value: unknown, size: number): boolean =>
  Array.isArray(value) && value.length === size && value.every(isFiniteNumber);

export function isFlowVariableOverrideValueCompatible(type: FlowValueType, value: unknown): boolean {
  switch (type) {
    case 'bool': return typeof value === 'boolean';
    case 'int': return Number.isInteger(value);
    case 'float': return isFiniteNumber(value);
    case 'vec2': return isVector(value, 2);
    case 'vec3': return isVector(value, 3);
    case 'vec4': return isVector(value, 4);
    case 'string':
    case 'name':
    case 'entity':
    case 'component': return typeof value === 'string';
    case 'any': return value !== undefined;
  }
}

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

export function reconcileFlowVariableOverridesWithDiagnostics(
  variables: readonly FlowVariableDefinition[],
  overrides: readonly FlowVariableOverride[],
): FlowVariableOverrideReconciliation {
  const variablesById = new Map(variables.map((variable) => [variable.id, variable]));
  const seen = new Set<string>();
  const reconciled: FlowVariableOverride[] = [];
  const diagnostics: FlowVariableOverrideDiagnostic[] = [];

  for (const override of overrides) {
    if (seen.has(override.variableId)) {
      diagnostics.push({ variableId: override.variableId, reason: 'duplicate' });
      continue;
    }
    seen.add(override.variableId);

    const variable = variablesById.get(override.variableId);
    if (!variable) {
      diagnostics.push({ variableId: override.variableId, reason: 'missing-variable' });
      continue;
    }
    if (!variable.exposed) {
      diagnostics.push({ variableId: override.variableId, reason: 'not-exposed' });
      continue;
    }
    if (!isFlowVariableOverrideValueCompatible(variable.type, override.value)) {
      diagnostics.push({ variableId: override.variableId, reason: 'invalid-value' });
      continue;
    }

    reconciled.push(override);
  }

  return { overrides: reconciled, diagnostics };
}

export function reconcileFlowVariableOverrides(
  variables: readonly FlowVariableDefinition[],
  overrides: readonly FlowVariableOverride[],
): FlowVariableOverride[] {
  return reconcileFlowVariableOverridesWithDiagnostics(variables, overrides).overrides;
}
