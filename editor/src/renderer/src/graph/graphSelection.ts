export type GraphSelectionId = string;

export interface GraphSelectionState {
  readonly ids: ReadonlySet<GraphSelectionId>;
  readonly anchor: GraphSelectionId | null;
}

export interface GraphClipboardNode<T> {
  readonly id: GraphSelectionId;
  readonly value: T;
}

export interface GraphClipboardSnapshot<T> {
  readonly nodes: readonly GraphClipboardNode<T>[];
}

export function createGraphSelection(
  ids: Iterable<GraphSelectionId> = [],
  anchor: GraphSelectionId | null = null,
): GraphSelectionState {
  return { ids: new Set(ids), anchor };
}

export function selectGraphNode(
  selection: GraphSelectionState,
  id: GraphSelectionId,
  additive = false,
): GraphSelectionState {
  if (!additive) return createGraphSelection([id], id);

  const ids = new Set(selection.ids);
  if (ids.has(id)) ids.delete(id);
  else ids.add(id);
  return { ids, anchor: id };
}

export function selectGraphRange(
  selection: GraphSelectionState,
  orderedIds: readonly GraphSelectionId[],
  id: GraphSelectionId,
  additive = false,
): GraphSelectionState {
  const anchor = selection.anchor ?? id;
  const from = orderedIds.indexOf(anchor);
  const to = orderedIds.indexOf(id);
  if (from < 0 || to < 0) return selectGraphNode(selection, id, additive);

  const ids = additive ? new Set(selection.ids) : new Set<GraphSelectionId>();
  const start = Math.min(from, to);
  const end = Math.max(from, to);
  for (let index = start; index <= end; index += 1) ids.add(orderedIds[index]);
  return { ids, anchor };
}

export function createGraphClipboardSnapshot<T>(
  nodes: readonly GraphClipboardNode<T>[],
  selection: GraphSelectionState,
): GraphClipboardSnapshot<T> {
  return { nodes: nodes.filter((node) => selection.ids.has(node.id)) };
}

export function remapGraphClipboardSnapshot<T>(
  snapshot: GraphClipboardSnapshot<T>,
  createId: (sourceId: GraphSelectionId, index: number) => GraphSelectionId,
): GraphClipboardSnapshot<T> {
  const seen = new Set<GraphSelectionId>();
  return {
    nodes: snapshot.nodes.map((node, index) => {
      const id = createId(node.id, index);
      if (seen.has(id)) throw new Error(`Duplicate graph node id generated while pasting: ${id}`);
      seen.add(id);
      return { ...node, id };
    }),
  };
}
