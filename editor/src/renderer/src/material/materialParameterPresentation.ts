import type { MaterialGraphNode, MaterialGraphParameter } from './materialGraphTypes';

export type MaterialParameterMetadata = MaterialGraphParameter & {
  group?: string;
  sortOrder?: number;
};

export type MaterialParameterPresentation = {
  nodeId: string;
  name: string;
  group: string;
  sortOrder: number;
};

export type MaterialParameterGroup = {
  name: string;
  parameters: MaterialParameterPresentation[];
};

export const defaultMaterialParameterGroup = 'Parameters';

const normalizedText = (value: string | undefined) => value?.trim() ?? '';

const normalizedSortOrder = (value: number | undefined) =>
  Number.isFinite(value) ? (value as number) : 0;

export const materialParameterPresentation = (
  node: MaterialGraphNode,
): MaterialParameterPresentation | null => {
  const parameter = node.parameter as MaterialParameterMetadata | undefined;
  if (!parameter?.exposed) {
    return null;
  }

  const name = normalizedText(parameter.name);
  if (!name) {
    return null;
  }

  return {
    nodeId: node.id,
    name,
    group: normalizedText(parameter.group) || defaultMaterialParameterGroup,
    sortOrder: normalizedSortOrder(parameter.sortOrder),
  };
};

export const groupMaterialParameters = (nodes: readonly MaterialGraphNode[]): MaterialParameterGroup[] => {
  const parameters = nodes
    .map(materialParameterPresentation)
    .filter((parameter): parameter is MaterialParameterPresentation => parameter !== null)
    .sort((left, right) => {
      const group = left.group.localeCompare(right.group);
      if (group !== 0) return group;
      const order = left.sortOrder - right.sortOrder;
      if (order !== 0) return order;
      const name = left.name.localeCompare(right.name);
      if (name !== 0) return name;
      return left.nodeId.localeCompare(right.nodeId);
    });

  const groups: MaterialParameterGroup[] = [];
  for (const parameter of parameters) {
    const previous = groups.at(-1);
    if (previous?.name === parameter.group) {
      previous.parameters.push(parameter);
    } else {
      groups.push({ name: parameter.group, parameters: [parameter] });
    }
  }
  return groups;
};
