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

export interface HierarchySearchEntity extends HierarchyMoveEntity {
  label: string;
  searchTerms?: readonly string[];
}

export interface HierarchySearchResult {
  visibleGuids: string[];
  matchedGuids: string[];
}

const bySiblingOrder = (a: HierarchyMoveEntity, b: HierarchyMoveEntity) =>
  a.siblingOrder - b.siblingOrder || a.guid.localeCompare(b.guid);

const normalizeSearch = (value: string) => value.trim().toLocaleLowerCase();

export const filterHierarchy = (entities: readonly HierarchySearchEntity[], query: string): HierarchySearchResult => {
  const normalizedQuery = normalizeSearch(query);
  if (!normalizedQuery) {
    return {
      visibleGuids: entities.map((entity) => entity.guid),
      matchedGuids: entities.map((entity) => entity.guid),
    };
  }

  const terms = normalizedQuery.split(/\s+/).filter(Boolean);
  const byGuid = new Map(entities.map((entity) => [entity.guid, entity]));
  const matched = new Set<string>();
  const visible = new Set<string>();

  for (const entity of entities) {
    const haystack = [entity.label, ...(entity.searchTerms ?? [])].map(normalizeSearch).filter(Boolean);
    if (!terms.every((term) => haystack.some((candidate) => candidate.includes(term)))) continue;

    matched.add(entity.guid);
    visible.add(entity.guid);

    let parentGuid = entity.parentGuid;
    const visited = new Set<string>();
    while (parentGuid && !visited.has(parentGuid)) {
      visited.add(parentGuid);
      visible.add(parentGuid);
      parentGuid = byGuid.get(parentGuid)?.parentGuid ?? '';
    }
  }

  return {
    visibleGuids: entities.filter((entity) => visible.has(entity.guid)).map((entity) => entity.guid),
    matchedGuids: entities.filter((entity) => matched.has(entity.guid)).map((entity) => entity.guid),
  };
};

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

  const destination = entities
    .filter((entity) => entity.guid !== guid && entity.parentGuid === parentGuid)
    .sort(bySiblingOrder);
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
