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

export interface GraphSelectionRect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface GraphSelectableBounds {
  readonly id: GraphSelectionId;
  readonly bounds: GraphSelectionRect;
}

export type GraphSelectionNavigationDirection = 'previous' | 'next' | 'first' | 'last';

export function createGraphSelection(
  ids: Iterable<GraphSelectionId> = [],
  anchor: GraphSelectionId | null = null,
): GraphSelectionState {
  return { ids: new Set(ids), anchor };
}

export function reconcileGraphSelection(
  selection: GraphSelectionState,
  orderedIds: readonly GraphSelectionId[],
): GraphSelectionState {
  const validIds = new Set(orderedIds);
  const ids = new Set([...selection.ids].filter((id) => validIds.has(id)));
  const anchor =
    selection.anchor !== null && validIds.has(selection.anchor)
      ? selection.anchor
      : (orderedIds.find((id) => ids.has(id)) ?? null);
  return { ids, anchor };
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

function normalizeRect(rect: GraphSelectionRect): GraphSelectionRect {
  return {
    x: rect.width < 0 ? rect.x + rect.width : rect.x,
    y: rect.height < 0 ? rect.y + rect.height : rect.y,
    width: Math.abs(rect.width),
    height: Math.abs(rect.height),
  };
}

function rectsIntersect(a: GraphSelectionRect, b: GraphSelectionRect): boolean {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  return right >= left && bottom >= top;
}

export function selectGraphMarquee(
  selection: GraphSelectionState,
  selectable: readonly GraphSelectableBounds[],
  marquee: GraphSelectionRect,
  additive = false,
): GraphSelectionState {
  const normalizedMarquee = normalizeRect(marquee);
  const ids = additive ? new Set(selection.ids) : new Set<GraphSelectionId>();
  let anchor = additive ? selection.anchor : null;

  for (const item of selectable) {
    if (!rectsIntersect(normalizedMarquee, normalizeRect(item.bounds))) continue;
    ids.add(item.id);
    anchor ??= item.id;
  }

  return { ids, anchor };
}

export function navigateGraphSelection(
  selection: GraphSelectionState,
  orderedIds: readonly GraphSelectionId[],
  direction: GraphSelectionNavigationDirection,
  extend = false,
): GraphSelectionState {
  if (orderedIds.length === 0) return selection;

  const current = selection.anchor === null ? -1 : orderedIds.indexOf(selection.anchor);
  let targetIndex: number;
  switch (direction) {
    case 'first':
      targetIndex = 0;
      break;
    case 'last':
      targetIndex = orderedIds.length - 1;
      break;
    case 'previous':
      targetIndex = current < 0 ? orderedIds.length - 1 : Math.max(0, current - 1);
      break;
    case 'next':
      targetIndex = current < 0 ? 0 : Math.min(orderedIds.length - 1, current + 1);
      break;
  }

  const target = orderedIds[targetIndex];
  if (!extend) return createGraphSelection([target], target);

  const anchor = selection.anchor !== null && current >= 0 ? selection.anchor : target;
  const anchorIndex = orderedIds.indexOf(anchor);
  const ids = new Set<GraphSelectionId>();
  const start = Math.min(anchorIndex, targetIndex);
  const end = Math.max(anchorIndex, targetIndex);
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
