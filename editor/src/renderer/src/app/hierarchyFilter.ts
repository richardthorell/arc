export type HierarchyFilterEntry = Readonly<{
  id: string;
  name: string;
  parentId: string | null;
}>;

export type HierarchyFilterResult = Readonly<{
  visibleIds: readonly string[];
  matchedIds: ReadonlySet<string>;
  expandedAncestorIds: ReadonlySet<string>;
}>;

function normalizeQuery(query: string): string {
  return query.trim().toLocaleLowerCase();
}

/**
 * Builds the visible hierarchy for a search query while preserving tree context.
 * Matching is intentionally presentation-only: entity IDs and authored expansion
 * state stay untouched, and ancestors are only expanded for the filtered view.
 */
export function filterHierarchy(entries: readonly HierarchyFilterEntry[], query: string): HierarchyFilterResult {
  const normalizedQuery = normalizeQuery(query);
  if (normalizedQuery.length === 0) {
    return {
      visibleIds: entries.map((entry) => entry.id),
      matchedIds: new Set(),
      expandedAncestorIds: new Set(),
    };
  }

  const byId = new Map(entries.map((entry) => [entry.id, entry] as const));
  const matchedIds = new Set<string>();
  const visibleIds = new Set<string>();
  const expandedAncestorIds = new Set<string>();

  for (const entry of entries) {
    if (!entry.name.toLocaleLowerCase().includes(normalizedQuery)) continue;

    matchedIds.add(entry.id);
    visibleIds.add(entry.id);

    const visited = new Set<string>([entry.id]);
    let parentId = entry.parentId;
    while (parentId != null && !visited.has(parentId)) {
      visited.add(parentId);
      const parent = byId.get(parentId);
      if (parent == null) break;
      visibleIds.add(parent.id);
      expandedAncestorIds.add(parent.id);
      parentId = parent.parentId;
    }
  }

  return {
    visibleIds: entries.filter((entry) => visibleIds.has(entry.id)).map((entry) => entry.id),
    matchedIds,
    expandedAncestorIds,
  };
}
