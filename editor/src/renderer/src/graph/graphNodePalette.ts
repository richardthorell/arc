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

export type GraphNodePaletteMovement = 'next' | 'previous' | 'first' | 'last';

function normalize(value: string): string {
  return value.trim().toLocaleLowerCase();
}

function searchableText(descriptor: GraphNodePaletteDescriptor): string {
  return [descriptor.name, descriptor.category ?? '', descriptor.description ?? '', ...(descriptor.keywords ?? [])]
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

/**
 * Resolves keyboard navigation for an already-filtered palette result list.
 *
 * Selection is identified by the domain-stable node kind instead of a list
 * index so changing the search query cannot accidentally activate a different
 * node. If the previous selection is no longer present, navigation restarts
 * from the appropriate edge of the current results.
 */
export function moveGraphNodePaletteSelection<TKind extends string>(
  descriptors: readonly GraphNodePaletteDescriptor<TKind>[],
  selectedKind: TKind | undefined,
  movement: GraphNodePaletteMovement,
): TKind | undefined {
  if (descriptors.length === 0) return undefined;
  if (movement === 'first') return descriptors[0].kind;
  if (movement === 'last') return descriptors[descriptors.length - 1].kind;

  const selectedIndex = selectedKind === undefined ? -1 : descriptors.findIndex(({ kind }) => kind === selectedKind);
  if (selectedIndex < 0) {
    return movement === 'previous' ? descriptors[descriptors.length - 1].kind : descriptors[0].kind;
  }

  const offset = movement === 'next' ? 1 : -1;
  const nextIndex = (selectedIndex + offset + descriptors.length) % descriptors.length;
  return descriptors[nextIndex].kind;
}

/** Returns the selected descriptor only when it still belongs to the current result set. */
export function resolveGraphNodePaletteSelection<TKind extends string>(
  descriptors: readonly GraphNodePaletteDescriptor<TKind>[],
  selectedKind: TKind | undefined,
): GraphNodePaletteDescriptor<TKind> | undefined {
  if (selectedKind === undefined) return undefined;
  return descriptors.find(({ kind }) => kind === selectedKind);
}
