export type HierarchyFilterEntry = Readonly<{
  id: string;
  parentId: string | null;
  name: string;
}>;

export type HierarchyFilterResult = Readonly<{
  visibleIds: ReadonlySet<string>;
  matchedIds: ReadonlySet<string>;
}>;

const normalizeQuery = (query: string): string => query.trim().toLocaleLowerCase();

/**
 * Computes hierarchy visibility for search without mutating expansion state.
 *
 * Direct matches remain distinguishable from ancestors that are only visible
 * to preserve context. Walking parent links instead of recursively scanning
 * children keeps the result deterministic for large, flat scene snapshots.
 */
export function filterHierarchy(
  entries: readonly HierarchyFilterEntry[],
  query: string,
): HierarchyFilterResult {
  const normalizedQuery = normalizeQuery(query);
  const allIds = new Set(entries.map((entry) => entry.id));
  if (!normalizedQuery) return { visibleIds: allIds, matchedIds: allIds };

  const byId = new Map(entries.map((entry) => [entry.id, entry] as const));
  const matchedIds = new Set(
    entries
      .filter((entry) => entry.name.toLocaleLowerCase().includes(normalizedQuery))
      .map((entry) => entry.id),
  );
  const visibleIds = new Set(matchedIds);

  for (const matchedId of matchedIds) {
    let parentId = byId.get(matchedId)?.parentId ?? null;
    const visited = new Set<string>();
    while (parentId && !visited.has(parentId)) {
      visited.add(parentId);
      const parent = byId.get(parentId);
      if (!parent) break;
      visibleIds.add(parent.id);
      parentId = parent.parentId;
    }
  }

  return { visibleIds, matchedIds };
}
