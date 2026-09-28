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

export type GraphClipboardPasteResult<Node extends GraphClipboardNode, Edge extends GraphClipboardEdge> =
  GraphClipboardPayload<Node, Edge> & {
    nodeIdMap: ReadonlyMap<string, string>;
    selectedNodeIds: ReadonlySet<string>;
  };

export type GraphIdFactory = (kind: 'node' | 'edge', previousId: string) => string;

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
): GraphClipboardPasteResult<Node, Edge> {
  const nodeIdMap = new Map<string, string>();
  const nodes = payload.nodes.map((node) => {
    const id = createId('node', node.id);
    nodeIdMap.set(node.id, id);
    return {
      ...structuredClone(node),
      id,
      position: { x: node.position.x + offset.x, y: node.position.y + offset.y },
    } as Node;
  });

  const edges = payload.edges.flatMap((edge) => {
    const sourceNodeId = nodeIdMap.get(edge.sourceNodeId);
    const targetNodeId = nodeIdMap.get(edge.targetNodeId);
    if (!sourceNodeId || !targetNodeId) return [];
    return [
      {
        ...structuredClone(edge),
        id: createId('edge', edge.id),
        sourceNodeId,
        targetNodeId,
      } as Edge,
    ];
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
  return pasteGraphSelection(copyGraphSelection(nodes, edges, selectedNodeIds), createId, offset);
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
