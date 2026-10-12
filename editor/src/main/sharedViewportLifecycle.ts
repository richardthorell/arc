/** Remove per-viewport bookkeeping for a window whose Electron objects may already be destroyed. */
export function forgetWindowViewportEntries(
  targetWebContentsId: number,
  targets: Map<string, number>,
  collections: ReadonlyArray<{ delete(key: string): boolean }>,
): void {
  for (const [viewportId, ownerId] of [...targets]) {
    if (ownerId !== targetWebContentsId) continue;
    targets.delete(viewportId);
    for (const collection of collections) collection.delete(viewportId);
  }
}
