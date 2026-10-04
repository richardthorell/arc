export type HierarchySelectionEntry = Readonly<{
  id: string;
  parentId: string | null;
  label: string;
}>;

export type HierarchySelectionSummary = Readonly<{
  count: number;
  validIds: readonly string[];
  staleIds: readonly string[];
  commonParentId: string | null;
  hasCommonParent: boolean;
  label: string;
}>;

/**
 * Builds presentation-only state for the hierarchy's current selection.
 *
 * Scene mutations can remove entities while selection state is still propagating,
 * so stale IDs are reported separately rather than being presented as valid
 * entities. The summary preserves authored hierarchy order for deterministic UI.
 */
export function summarizeHierarchySelection(
  entries: readonly HierarchySelectionEntry[],
  selectedIds: readonly string[],
): HierarchySelectionSummary {
  const byId = new Map(entries.map((entry) => [entry.id, entry] as const));
  const selectedSet = new Set(selectedIds);
  const valid = entries.filter((entry) => selectedSet.has(entry.id));
  const staleIds = selectedIds.filter((id) => !byId.has(id));

  const commonParentId = valid[0]?.parentId ?? null;
  const hasCommonParent =
    valid.length > 0 && valid.every((entry) => entry.parentId === commonParentId);

  let label = 'No selection';
  if (valid.length === 1) {
    label = valid[0]?.label ?? '1 entity';
  } else if (valid.length > 1) {
    label = `${valid.length} entities`;
  }

  return {
    count: valid.length,
    validIds: valid.map((entry) => entry.id),
    staleIds,
    commonParentId,
    hasCommonParent,
    label,
  };
}
