export type HierarchySelection = Readonly<{
  ids: ReadonlySet<string>
  anchorId: string | null
}>

export type HierarchySelectionIntent = 'replace' | 'toggle' | 'range'

export function updateHierarchySelection(
  selection: HierarchySelection,
  visibleIds: readonly string[],
  targetId: string,
  intent: HierarchySelectionIntent,
): HierarchySelection {
  if (!visibleIds.includes(targetId)) return selection

  if (intent === 'replace') {
    return { ids: new Set([targetId]), anchorId: targetId }
  }

  if (intent === 'toggle') {
    const ids = new Set(selection.ids)
    if (ids.has(targetId)) ids.delete(targetId)
    else ids.add(targetId)
    return { ids, anchorId: targetId }
  }

  const anchorId = selection.anchorId
  const anchorIndex = anchorId == null ? -1 : visibleIds.indexOf(anchorId)
  const targetIndex = visibleIds.indexOf(targetId)
  if (anchorIndex < 0) {
    return { ids: new Set([targetId]), anchorId: targetId }
  }

  const first = Math.min(anchorIndex, targetIndex)
  const last = Math.max(anchorIndex, targetIndex)
  return { ids: new Set(visibleIds.slice(first, last + 1)), anchorId }
}

export function pruneHierarchySelection(
  selection: HierarchySelection,
  existingIds: ReadonlySet<string>,
): HierarchySelection {
  const ids = new Set([...selection.ids].filter((id) => existingIds.has(id)))
  const anchorId = selection.anchorId != null && existingIds.has(selection.anchorId) ? selection.anchorId : null
  return { ids, anchorId }
}
