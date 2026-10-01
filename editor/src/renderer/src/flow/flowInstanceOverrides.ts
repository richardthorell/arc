import type { FlowGraph, FlowValueType, FlowVariableDefinition } from './flowGraphTypes';

export type FlowInstanceOverride = {
  variableId: string;
  value: unknown;
};

export type FlowOverrideDiagnostic = {
  variableId: string;
  variableName?: string;
  reason: 'missing-variable' | 'not-exposed' | 'type-mismatch';
  message: string;
};

export type FlowOverrideReconciliation = {
  overrides: FlowInstanceOverride[];
  diagnostics: FlowOverrideDiagnostic[];
};

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const isNumericTuple = (value: unknown, size: number) =>
  Array.isArray(value) && value.length === size && value.every(isFiniteNumber);

export const isFlowOverrideValueCompatible = (type: FlowValueType, value: unknown): boolean => {
  switch (type) {
    case 'bool':
      return typeof value === 'boolean';
    case 'int':
      return Number.isInteger(value);
    case 'float':
      return isFiniteNumber(value);
    case 'vec2':
      return isNumericTuple(value, 2);
    case 'vec3':
      return isNumericTuple(value, 3);
    case 'vec4':
      return isNumericTuple(value, 4);
    case 'string':
    case 'name':
    case 'entity':
    case 'component':
      return typeof value === 'string';
    case 'any':
      return true;
  }
};

export const getExposedFlowVariables = (graph: FlowGraph): FlowVariableDefinition[] =>
  graph.variables.filter((variable) => variable.exposed);

export const reconcileFlowInstanceOverrides = (
  graph: FlowGraph,
  overrides: readonly FlowInstanceOverride[],
): FlowOverrideReconciliation => {
  const variablesById = new Map(graph.variables.map((variable) => [variable.id, variable]));
  const reconciled: FlowInstanceOverride[] = [];
  const diagnostics: FlowOverrideDiagnostic[] = [];
  const seen = new Set<string>();

  for (const override of overrides) {
    if (seen.has(override.variableId)) continue;
    seen.add(override.variableId);

    const variable = variablesById.get(override.variableId);
    if (!variable) {
      diagnostics.push({
        variableId: override.variableId,
        reason: 'missing-variable',
        message: `Flow override references missing variable ${override.variableId}`,
      });
      continue;
    }
    if (!variable.exposed) {
      diagnostics.push({
        variableId: variable.id,
        variableName: variable.name,
        reason: 'not-exposed',
        message: `Flow variable ${variable.name} is no longer exposed`,
      });
      continue;
    }
    if (!isFlowOverrideValueCompatible(variable.type, override.value)) {
      diagnostics.push({
        variableId: variable.id,
        variableName: variable.name,
        reason: 'type-mismatch',
        message: `Flow override for ${variable.name} is incompatible with ${variable.type}`,
      });
      continue;
    }
    reconciled.push({ variableId: override.variableId, value: override.value });
  }

  return { overrides: reconciled, diagnostics };
};
