import type { LivePlayEntity, LivePlayMessage } from '../../../common/livePlayProtocol';

export type RuntimeHierarchyMessage = Extract<LivePlayMessage, { type: 'hierarchy.snapshot' }>;
export type RuntimeInspectionResult = Extract<LivePlayMessage, { type: 'component.inspect.result' }>['payload'];
export type RuntimeHierarchyItem = RuntimeHierarchyMessage['payload']['entities'][number];

/** Runtime-only IDs and authored GUIDs may share text, but never an identity namespace. */
export function runtimeEntityKey(entity: LivePlayEntity): string {
  return entity.kind === 'authored' ? `authored:${entity.entityGuid}` : `runtime:${entity.runtimeEntityId}`;
}

export type RuntimeHierarchyRow = RuntimeHierarchyItem & { key: string; depth: number };

/** Preserve native order and bound traversal for malformed/cyclic parent graphs. */
export function runtimeHierarchyRows(entities: readonly RuntimeHierarchyItem[]): RuntimeHierarchyRow[] {
  const byKey = new Map(entities.map((item) => [runtimeEntityKey(item.entity), item]));
  return entities.map((item) => {
    const key = runtimeEntityKey(item.entity);
    const visited = new Set([key]);
    let depth = 0;
    let parent = item.parent;
    while (parent && depth < 16) {
      const parentKey = runtimeEntityKey(parent);
      if (visited.has(parentKey) || !byKey.has(parentKey)) break;
      visited.add(parentKey);
      depth += 1;
      parent = byKey.get(parentKey)?.parent ?? null;
    }
    return { ...item, key, depth };
  });
}

export function matchesRuntimeEntity(left: LivePlayEntity | null, right: LivePlayEntity): boolean {
  return left !== null && runtimeEntityKey(left) === runtimeEntityKey(right);
}

/** Never display an older out-of-order result over a newer runtime revision. */
export function chooseRuntimeInspection(
  previous: RuntimeInspectionResult | undefined,
  incoming: RuntimeInspectionResult,
): RuntimeInspectionResult {
  return previous && previous.runtimeRevision > incoming.runtimeRevision ? previous : incoming;
}
