export type HierarchyGroupEntry = Readonly<{
  id: string;
  parentId: string | null;
}>;

export type HierarchyGroupPlan = Readonly<{
  parentId: string | null;
  memberIds: readonly string[];
}>;

/**
 * Builds the structural portion of a hierarchy grouping operation.
 *
 * Grouping is only valid when every selected entity is a sibling. Keeping this
 * validation separate from presentation lets the command/undo layer create the
 * actual group entity without the hierarchy view inventing scene mutations.
 */
export function planHierarchyGroup(
  entries: readonly HierarchyGroupEntry[],
  selectedIds: readonly string[],
): HierarchyGroupPlan | null {
  if (selectedIds.length < 2) return null;

  const byId = new Map(entries.map((entry) => [entry.id, entry] as const));
  const uniqueSelection = new Set(selectedIds);
  if (uniqueSelection.size !== selectedIds.length) return null;

  const selected = selectedIds.map((id) => byId.get(id));
  if (selected.some((entry) => entry == null)) return null;

  const parentId = selected[0]?.parentId ?? null;
  if (selected.some((entry) => entry?.parentId !== parentId)) return null;

  const selectedSet = new Set(selectedIds);
  const memberIds = entries.filter((entry) => selectedSet.has(entry.id)).map((entry) => entry.id);

  return { parentId, memberIds };
}
