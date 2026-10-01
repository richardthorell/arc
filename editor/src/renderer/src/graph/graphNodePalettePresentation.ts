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
  const groups: GraphNodePaletteGroup<TKind>[] = [];

  for (const descriptor of results) {
    const category = descriptor.category?.trim() || DEFAULT_CATEGORY;
    const previous = groups[groups.length - 1];
    if (previous?.category === category) {
      previous.nodes.push(descriptor);
    } else {
      groups.push({ category, nodes: [descriptor] });
    }
  }

  const selectionIsVisible = selectedKind !== undefined && results.some(({ kind }) => kind === selectedKind);

  return {
    groups,
    resultCount: results.length,
    selectedKind: selectionIsVisible ? selectedKind : results[0]?.kind,
  };
}
