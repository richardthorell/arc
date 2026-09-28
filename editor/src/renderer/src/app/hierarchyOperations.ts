export interface HierarchyMoveEntity {
  guid: string;
  parentGuid: string;
  siblingOrder: number;
}

export interface HierarchyMove {
  guid: string;
  parentGuid: string;
  siblingOrder: number;
}

export interface HierarchyMovePlan {
  moved: HierarchyMove;
  siblings: HierarchyMove[];
}

const bySiblingOrder = (a: HierarchyMoveEntity, b: HierarchyMoveEntity) =>
  a.siblingOrder - b.siblingOrder || a.guid.localeCompare(b.guid);

export const planHierarchyMove = (
  entities: readonly HierarchyMoveEntity[],
  guid: string,
  parentGuid: string,
  siblingOrder: number,
): HierarchyMovePlan | null => {
  const byGuid = new Map(entities.map((entity) => [entity.guid, entity]));
  const moved = byGuid.get(guid);
  if (!moved || guid === parentGuid || (parentGuid && !byGuid.has(parentGuid))) return null;

  let ancestorGuid = parentGuid;
  const visited = new Set<string>();
  while (ancestorGuid) {
    if (ancestorGuid === guid || visited.has(ancestorGuid)) return null;
    visited.add(ancestorGuid);
    ancestorGuid = byGuid.get(ancestorGuid)?.parentGuid ?? '';
  }

  const destination = entities.filter((entity) => entity.guid !== guid && entity.parentGuid === parentGuid).sort(bySiblingOrder);
  const insertionIndex = Math.max(0, Math.min(Math.trunc(siblingOrder), destination.length));
  destination.splice(insertionIndex, 0, moved);

  const siblings = destination.map((entity, index) => ({
    guid: entity.guid,
    parentGuid,
    siblingOrder: index,
  }));

  return {
    moved: siblings.find((entity) => entity.guid === guid)!,
    siblings,
  };
};
