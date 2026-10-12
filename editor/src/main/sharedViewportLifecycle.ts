/** Remove per-viewport bookkeeping for a window whose Electron objects may already be destroyed. */
export function forgetWindowViewportEntries<T>(
  targetWebContentsId: number,
  targets: Map<string, number>,
  collections: ReadonlyArray<Map<string, T> | Set<string>>,
): void {
  for (const [viewportId, ownerId] of [...targets]) {
    if (ownerId !== targetWebContentsId) continue;
    targets.delete(viewportId);
    for (const collection of collections) collection.delete(viewportId);
  }
}
