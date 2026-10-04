export type GraphClipboardNode = {
  id: string;
  position: { x: number; y: number };
};

export type GraphClipboardEdge = {
  id: string;
  sourceNodeId: string;
  targetNodeId: string;
};

export type GraphClipboardPayload<Node extends GraphClipboardNode, Edge extends GraphClipboardEdge> = {
  nodes: Node[];
  edges: Edge[];
};

export type GraphClipboardPasteResult<
  Node extends GraphClipboardNode,
  Edge extends GraphClipboardEdge,
> = GraphClipboardPayload<Node, Edge> & {
  nodeIdMap: ReadonlyMap<string, string>;
  selectedNodeIds: ReadonlySet<string>;
};

export type GraphIdFactory = (kind: 'node' | 'edge', previousId: string) => string;

function assertUniqueIds(items: readonly { id: string }[], kind: 'node' | 'edge'): void {
  const ids = new Set<string>();
  for (const item of items) {
    if (!item.id) throw new Error(`Graph clipboard ${kind} IDs must not be empty`);
    if (ids.has(item.id)) throw new Error(`Graph clipboard contains duplicate ${kind} ID: ${item.id}`);
    ids.add(item.id);
  }
}

function assertValidClipboardPayload<Node extends GraphClipboardNode, Edge extends GraphClipboardEdge>(
  payload: GraphClipboardPayload<Node, Edge>,
): void {
  assertUniqueIds(payload.nodes, 'node');
  assertUniqueIds(payload.edges, 'edge');

  const nodeIds = new Set(payload.nodes.map((node) => node.id));
  for (const edge of payload.edges) {
    if (!nodeIds.has(edge.sourceNodeId) || !nodeIds.has(edge.targetNodeId)) {
      throw new Error(`Graph clipboard edge ${edge.id} references a node outside the clipboard payload`);
    }
  }
}

export function copyGraphSelection<Node extends GraphClipboardNode, Edge extends GraphClipboardEdge>(
  nodes: readonly Node[],
  edges: readonly Edge[],
  selectedNodeIds: ReadonlySet<string>,
): GraphClipboardPayload<Node, Edge> {
  const copiedNodes = nodes.filter((node) => selectedNodeIds.has(node.id));
  const copiedIds = new Set(copiedNodes.map((node) => node.id));
  return {
    nodes: copiedNodes.map((node) => structuredClone(node)),
    edges: edges
      .filter((edge) => copiedIds.has(edge.sourceNodeId) && copiedIds.has(edge.targetNodeId))
      .map((edge) => structuredClone(edge)),
  };
}

export function pasteGraphSelection<Node extends GraphClipboardNode, Edge extends GraphClipboardEdge>(
  payload: GraphClipboardPayload<Node, Edge>,
  createId: GraphIdFactory,
  offset = { x: 24, y: 24 },
  occupiedIds: ReadonlySet<string> = new Set(),
): GraphClipboardPasteResult<Node, Edge> {
  assertValidClipboardPayload(payload);

  const generatedIds = new Set<string>();
  const nodeIdMap = new Map<string, string>();
  const claimGeneratedId = (id: string, kind: 'node' | 'edge'): void => {
    if (!id || generatedIds.has(id) || occupiedIds.has(id)) {
      throw new Error(`Graph clipboard generated unavailable ${kind} ID: ${id}`);
    }
    generatedIds.add(id);
  };

  const nodes = payload.nodes.map((node) => {
    const id = createId('node', node.id);
    claimGeneratedId(id, 'node');
    nodeIdMap.set(node.id, id);
    return {
      ...structuredClone(node),
      id,
      position: { x: node.position.x + offset.x, y: node.position.y + offset.y },
    } as Node;
  });

  const edges = payload.edges.map((edge) => {
    const sourceNodeId = nodeIdMap.get(edge.sourceNodeId);
    const targetNodeId = nodeIdMap.get(edge.targetNodeId);
    if (!sourceNodeId || !targetNodeId) {
      throw new Error(`Graph clipboard edge ${edge.id} could not be remapped`);
    }
    const id = createId('edge', edge.id);
    claimGeneratedId(id, 'edge');
    return {
      ...structuredClone(edge),
      id,
      sourceNodeId,
      targetNodeId,
    } as Edge;
  });

  return {
    nodes,
    edges,
    nodeIdMap,
    selectedNodeIds: new Set(nodes.map((node) => node.id)),
  };
}

export function duplicateGraphSelection<Node extends GraphClipboardNode, Edge extends GraphClipboardEdge>(
  nodes: readonly Node[],
  edges: readonly Edge[],
  selectedNodeIds: ReadonlySet<string>,
  createId: GraphIdFactory,
  offset = { x: 24, y: 24 },
): GraphClipboardPasteResult<Node, Edge> {
  const occupiedIds = new Set([...nodes.map((node) => node.id), ...edges.map((edge) => edge.id)]);
  return pasteGraphSelection(
    copyGraphSelection(nodes, edges, selectedNodeIds),
    createId,
    offset,
    occupiedIds,
  );
}

export function deleteGraphSelection<Node extends GraphClipboardNode, Edge extends GraphClipboardEdge>(
  nodes: readonly Node[],
  edges: readonly Edge[],
  selectedNodeIds: ReadonlySet<string>,
): GraphClipboardPayload<Node, Edge> {
  return {
    nodes: nodes.filter((node) => !selectedNodeIds.has(node.id)).map((node) => structuredClone(node)),
    edges: edges
      .filter((edge) => !selectedNodeIds.has(edge.sourceNodeId) && !selectedNodeIds.has(edge.targetNodeId))
      .map((edge) => structuredClone(edge)),
  };
}
