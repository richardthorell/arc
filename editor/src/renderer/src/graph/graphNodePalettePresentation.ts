import { queryGraphNodePalette, type GraphNodePaletteDescriptor, type GraphNodePaletteQuery } from './graphNodePalette';

export interface GraphNodePaletteGroup<TKind extends string = string> {
  category: string;
  nodes: GraphNodePaletteDescriptor<TKind>[];
}

export interface GraphNodePalettePresentation<TKind extends string = string> {
  groups: GraphNodePaletteGroup<TKind>[];
  resultCount: number;
  selectedKind?: TKind;
}

const DEFAULT_CATEGORY = 'Other';

function comparePaletteGroups<TKind extends string>(
  left: GraphNodePaletteGroup<TKind>,
  right: GraphNodePaletteGroup<TKind>,
): number {
  if (left.category === DEFAULT_CATEGORY) return right.category === DEFAULT_CATEGORY ? 0 : 1;
  if (right.category === DEFAULT_CATEGORY) return -1;
  return left.category.localeCompare(right.category);
}

/**
 * Builds the domain-neutral presentation state consumed by graph node palettes.
 * Search and pin compatibility stay centralized in queryGraphNodePalette while
 * domains only provide descriptors and compatibility predicates.
 */
export function buildGraphNodePalettePresentation<TKind extends string>(
  descriptors: readonly GraphNodePaletteDescriptor<TKind>[],
  query: GraphNodePaletteQuery = {},
  selectedKind?: TKind,
): GraphNodePalettePresentation<TKind> {
  const results = queryGraphNodePalette(descriptors, query);
  const groupedNodes = new Map<string, GraphNodePaletteDescriptor<TKind>[]>();

  for (const descriptor of results) {
    const category = descriptor.category?.trim() || DEFAULT_CATEGORY;
    const nodes = groupedNodes.get(category);
    if (nodes) {
      nodes.push(descriptor);
    } else {
      groupedNodes.set(category, [descriptor]);
    }
  }

  const groups = [...groupedNodes.entries()]
    .map(([category, nodes]) => ({ category, nodes }))
    .sort(comparePaletteGroups);
  const firstVisibleKind = groups[0]?.nodes[0]?.kind;
  const selectionIsVisible = selectedKind !== undefined && results.some(({ kind }) => kind === selectedKind);

  return {
    groups,
    resultCount: results.length,
    selectedKind: selectionIsVisible ? selectedKind : firstVisibleKind,
  };
}
