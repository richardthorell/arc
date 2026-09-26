export interface GraphNodePalettePinContext {
  direction: 'input' | 'output';
  type: string;
}

export interface GraphNodePaletteDescriptor<TKind extends string = string> {
  kind: TKind;
  name: string;
  category?: string;
  keywords?: readonly string[];
  description?: string;
  /** Domain-owned compatibility predicate. Shared palette code never interprets pin types. */
  isCompatible?: (context: GraphNodePalettePinContext) => boolean;
}

export interface GraphNodePaletteQuery {
  search?: string;
  pin?: GraphNodePalettePinContext;
}

function normalize(value: string): string {
  return value.trim().toLocaleLowerCase();
}

function searchableText(descriptor: GraphNodePaletteDescriptor): string {
  return [
    descriptor.name,
    descriptor.category ?? '',
    descriptor.description ?? '',
    ...(descriptor.keywords ?? []),
  ]
    .map(normalize)
    .join(' ');
}

/**
 * Filters and deterministically orders domain-provided node descriptors.
 *
 * Domains own node semantics and compatibility; this helper owns the common
 * palette mechanics so Material, Flow, and future graph editors do not fork
 * search behavior.
 */
export function queryGraphNodePalette<TKind extends string>(
  descriptors: readonly GraphNodePaletteDescriptor<TKind>[],
  query: GraphNodePaletteQuery = {},
): GraphNodePaletteDescriptor<TKind>[] {
  const terms = normalize(query.search ?? '')
    .split(/\s+/)
    .filter(Boolean);

  return descriptors
    .filter((descriptor) => {
      if (query.pin && descriptor.isCompatible && !descriptor.isCompatible(query.pin)) {
        return false;
      }
      if (query.pin && !descriptor.isCompatible) {
        return false;
      }

      const haystack = searchableText(descriptor);
      return terms.every((term) => haystack.includes(term));
    })
    .sort((left, right) => {
      const category = (left.category ?? '').localeCompare(right.category ?? '');
      if (category !== 0) return category;
      const name = left.name.localeCompare(right.name);
      if (name !== 0) return name;
      return left.kind.localeCompare(right.kind);
    });
}
