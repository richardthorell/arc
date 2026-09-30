import type { MaterialGraph } from './materialGraphTypes';
import { materialEditorParameters, type MaterialEditorParameter } from './materialCompiler';

export type MaterialParameterAuthoringMetadata = {
  group: string;
  description?: string;
  sortOrder: number;
};

export type MaterialAuthoringParameter = MaterialEditorParameter & MaterialParameterAuthoringMetadata;

const DEFAULT_GROUP = 'Parameters';

const readString = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim().length > 0 ? value.trim() : undefined;

const readSortOrder = (value: unknown): number => (typeof value === 'number' && Number.isFinite(value) ? value : 0);

/**
 * Build deterministic editor presentation metadata for exposed material parameters.
 *
 * Authoring metadata deliberately stays outside native compiler semantics: group, description,
 * and ordering affect presentation only. The fields live beside `exposed`/`name` on the authored
 * node parameter object, so normal graph JSON save/load preserves them without a parallel store.
 */
export const materialAuthoringParameters = (graph: MaterialGraph): MaterialAuthoringParameter[] => {
  const presentationByNode = new Map(materialEditorParameters(graph).map((parameter) => [parameter.nodeId, parameter]));

  return graph.nodes
    .flatMap((node) => {
      const parameter = presentationByNode.get(node.id);
      if (!parameter || !node.parameter) return [];

      const metadata = node.parameter as typeof node.parameter & Record<string, unknown>;
      return [
        {
          ...parameter,
          group: readString(metadata.group) ?? DEFAULT_GROUP,
          description: readString(metadata.description),
          sortOrder: readSortOrder(metadata.sortOrder),
        },
      ];
    })
    .sort(
      (left, right) =>
        left.group.localeCompare(right.group) ||
        left.sortOrder - right.sortOrder ||
        left.name.localeCompare(right.name) ||
        left.nodeId.localeCompare(right.nodeId),
    );
};

export const materialParameterGroups = (
  graph: MaterialGraph,
): Array<{ name: string; parameters: MaterialAuthoringParameter[] }> => {
  const groups = new Map<string, MaterialAuthoringParameter[]>();
  for (const parameter of materialAuthoringParameters(graph)) {
    const parameters = groups.get(parameter.group) ?? [];
    parameters.push(parameter);
    groups.set(parameter.group, parameters);
  }
  return [...groups].map(([name, parameters]) => ({ name, parameters }));
};
