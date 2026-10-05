export type HierarchySelection = ReadonlyArray<string>;

function unique(ids: Iterable<string>): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const id of ids) {
    if (!id || seen.has(id)) continue;
    seen.add(id);
    result.push(id);
  }
  return result;
}

/**
 * Applies desktop-style hierarchy selection semantics without coupling the
 * hierarchy UI to scene mutation or undo infrastructure.
 *
 * `visibleIds` must be the current flattened, filtered hierarchy order. Range
 * selection therefore remains predictable when branches are collapsed or a
 * search filter is active.
 */
export function updateHierarchySelection(
  selection: HierarchySelection,
  targetId: string,
  visibleIds: ReadonlyArray<string>,
  options: { toggle?: boolean; range?: boolean; anchorId?: string | null } = {},
): string[] {
  if (!targetId) return unique(selection);

  const current = unique(selection);
  if (options.range && options.anchorId) {
    const anchorIndex = visibleIds.indexOf(options.anchorId);
    const targetIndex = visibleIds.indexOf(targetId);
    if (anchorIndex >= 0 && targetIndex >= 0) {
      const start = Math.min(anchorIndex, targetIndex);
      const end = Math.max(anchorIndex, targetIndex);
      const range = visibleIds.slice(start, end + 1);
      return options.toggle ? unique([...current, ...range]) : unique(range);
    }
  }

  if (options.toggle) {
    return current.includes(targetId)
      ? current.filter((id) => id !== targetId)
      : [...current, targetId];
  }

  return [targetId];
}

/** Removes stale entities while preserving the user's selection order. */
export function reconcileHierarchySelection(
  selection: HierarchySelection,
  existingIds: Iterable<string>,
): string[] {
  const existing = new Set(existingIds);
  return unique(selection).filter((id) => existing.has(id));
}
