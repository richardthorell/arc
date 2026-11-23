import type { LivePlayEntity, LivePlayMessage } from './livePlayProtocol';

export type HierarchySnapshotMessage = Extract<LivePlayMessage, { type: 'hierarchy.snapshot' }>;
export type HierarchyChangeMessage = Extract<LivePlayMessage, { type: 'hierarchy.changed' }>;
export type HierarchyItem = HierarchySnapshotMessage['payload']['entities'][number];

type PendingSnapshot = {
  sessionId: string;
  instanceId: string;
  revision: number;
  pageCount: number;
  complete: boolean;
  pages: (HierarchyItem[] | null)[];
};

/** Pure, session-owned state. No partial page or unverified delta is displayed. */
export type LivePlayHierarchyState = {
  hierarchy: HierarchySnapshotMessage | null;
  pending: PendingSnapshot | null;
  requiresSnapshot: boolean;
};

export const emptyLivePlayHierarchy = (): LivePlayHierarchyState => ({
  hierarchy: null,
  pending: null,
  requiresSnapshot: false,
});

const maxHierarchyEntities = 4096;
const maxPageEntities = 64;

function entityKey(entity: LivePlayEntity): string {
  return entity.kind === 'authored' ? `authored:${entity.entityGuid}` : `runtime:${entity.runtimeEntityId}`;
}

function validEntitySet(items: readonly HierarchyItem[]): boolean {
  const keys = new Set<string>();
  for (const item of items) {
    const key = entityKey(item.entity);
    if (keys.has(key)) return false;
    keys.add(key);
  }
  return true;
}

/** Revisions are scoped to the exact native session + runtime instance. */
export function applyLivePlayHierarchy(
  previous: LivePlayHierarchyState,
  message: HierarchySnapshotMessage | HierarchyChangeMessage,
): LivePlayHierarchyState {
  const current = previous.hierarchy;
  if (
    current &&
    (current.sessionId !== message.sessionId || current.instanceId !== message.instanceId)
  ) return { hierarchy: null, pending: null, requiresSnapshot: true };

  if (message.type === 'hierarchy.changed') {
    const { baseRevision, worldRevision, changes } = message.payload;
    if (current && worldRevision <= current.payload.worldRevision) return previous;
    if (
      previous.pending ||
      !current ||
      !current.payload.complete ||
      previous.requiresSnapshot ||
      baseRevision === undefined ||
      baseRevision !== current.payload.worldRevision ||
      worldRevision <= baseRevision
    ) return { ...previous, pending: null, requiresSnapshot: true };

    const items = new Map(current.payload.entities.map((item) => [entityKey(item.entity), item]));
    for (const change of changes) {
      if (change.kind === 'removed') items.delete(entityKey(change.entity));
      else items.set(entityKey(change.value.entity), change.value);
    }
    const entities = Array.from(items.values());
    if (entities.length > maxHierarchyEntities || !validEntitySet(entities))
      return { ...previous, requiresSnapshot: true };
    return {
      hierarchy: {
        ...message,
        type: 'hierarchy.snapshot',
        payload: { worldRevision, complete: true, entities },
      },
      pending: null,
      requiresSnapshot: false,
    };
  }

  const { worldRevision, entities, complete, pageIndex, pageCount } = message.payload;
  if (current && worldRevision < current.payload.worldRevision) return previous;
  if (pageIndex === undefined && pageCount === undefined) {
    if (current && worldRevision === current.payload.worldRevision) return previous;
    if (!validEntitySet(entities)) return { ...previous, pending: null, requiresSnapshot: true };
    return {
      hierarchy: message,
      pending: null,
      requiresSnapshot: false,
    };
  }

  if (
    pageIndex === undefined ||
    pageCount === undefined ||
    pageIndex >= pageCount ||
    entities.length > maxPageEntities ||
    pageCount * maxPageEntities > maxHierarchyEntities
  ) return { ...previous, pending: null, requiresSnapshot: true };

  if (current && worldRevision === current.payload.worldRevision) return previous;
  const original = previous.pending;
  const pending: PendingSnapshot =
    original && original.sessionId === message.sessionId && original.instanceId === message.instanceId &&
    original.revision === worldRevision && original.pageCount === pageCount && original.complete === complete
      ? { ...original, pages: [...original.pages] }
      : {
          sessionId: message.sessionId,
          instanceId: message.instanceId,
          revision: worldRevision,
          pageCount,
          complete,
          pages: Array.from({ length: pageCount }, () => null),
        };
  if (pending.pages[pageIndex] !== null) return previous;
  pending.pages[pageIndex] = entities;
  if (pending.pages.some((page) => page === null))
    return { ...previous, pending, requiresSnapshot: previous.requiresSnapshot };

  const assembled = pending.pages.flatMap((page) => page ?? []);
  if (assembled.length > maxHierarchyEntities || !validEntitySet(assembled))
    return { ...previous, pending: null, requiresSnapshot: true };
  return {
    hierarchy: {
      ...message,
      type: 'hierarchy.snapshot',
      payload: { worldRevision, entities: assembled, complete },
    },
    pending: null,
    requiresSnapshot: false,
  };
}
