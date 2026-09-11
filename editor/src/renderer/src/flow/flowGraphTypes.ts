import type { GraphConnectionLike, GraphNodeDefinition, GraphNodeLike, GraphPoint, GraphViewport } from '../graph';

export type FlowValueType =
  'bool' | 'int' | 'float' | 'vec2' | 'vec3' | 'vec4' | 'string' | 'name' | 'entity' | 'component' | 'any';

export type FlowPinType =
  | { kind: 'execution' }
  | {
      kind: 'value';
      valueType: FlowValueType;
    };

export type FlowNodeType = 'beginPlay' | 'endPlay' | 'tick' | 'fixedTick' | 'inputAction' | 'branch';

export type FlowNodeCategory = 'Events' | 'Input' | 'Flow Control';
export type FlowNodeSubcategory = 'Lifecycle' | 'Update' | 'Actions' | 'Branching';

export type FlowGraphNode = GraphNodeLike<FlowNodeType> & {
  values: Record<string, unknown>;
};

export type FlowGraphConnection = GraphConnectionLike & {
  kind: 'execution' | 'value';
};

export type FlowVariableDefinition = {
  id: string;
  name: string;
  type: FlowValueType;
  defaultValue: unknown;
  exposed: boolean;
};

export type FlowGraph = {
  version: 1;
  variables: FlowVariableDefinition[];
  nodes: FlowGraphNode[];
  connections: FlowGraphConnection[];
  viewport: GraphViewport;
};

export type FlowAssetJson = {
  version: 1;
  assetType: 'flow';
  name: string;
  graph: FlowGraph;
};

const execution = (id: string, label: string) => ({ id, label, type: { kind: 'execution' } as FlowPinType });
const value = (id: string, label: string, valueType: FlowValueType) => ({
  id,
  label,
  type: { kind: 'value', valueType } as FlowPinType,
});

export const flowNodeDefinitions: Record<
  FlowNodeType,
  GraphNodeDefinition<FlowNodeType, FlowPinType, FlowNodeCategory, FlowNodeSubcategory>
> = {
  beginPlay: {
    type: 'beginPlay',
    title: 'Begin Play',
    category: 'Events',
    subcategory: 'Lifecycle',
    inputs: [],
    outputs: [execution('exec', 'Then')],
  },
  endPlay: {
    type: 'endPlay',
    title: 'End Play',
    category: 'Events',
    subcategory: 'Lifecycle',
    inputs: [],
    outputs: [execution('exec', 'Then')],
  },
  tick: {
    type: 'tick',
    title: 'Tick',
    category: 'Events',
    subcategory: 'Update',
    inputs: [],
    outputs: [execution('exec', 'Then'), value('deltaSeconds', 'Delta Seconds', 'float')],
  },
  fixedTick: {
    type: 'fixedTick',
    title: 'Fixed Tick',
    category: 'Events',
    subcategory: 'Update',
    inputs: [],
    outputs: [execution('exec', 'Then'), value('deltaSeconds', 'Delta Seconds', 'float')],
  },
  inputAction: {
    type: 'inputAction',
    title: 'Input Action',
    category: 'Input',
    subcategory: 'Actions',
    inputs: [],
    outputs: [
      execution('triggered', 'Triggered'),
      execution('completed', 'Completed'),
      value('value', 'Value', 'float'),
    ],
  },
  branch: {
    type: 'branch',
    title: 'Branch',
    category: 'Flow Control',
    subcategory: 'Branching',
    inputs: [execution('exec', 'In'), value('condition', 'Condition', 'bool')],
    outputs: [execution('true', 'True'), execution('false', 'False')],
  },
};

let generatedId = 0;
export const flowGraphId = (prefix: string) => `${prefix}-${Date.now().toString(36)}-${(generatedId++).toString(36)}`;

export const cloneFlowGraph = (graph: FlowGraph): FlowGraph => JSON.parse(JSON.stringify(graph)) as FlowGraph;

export const createFlowNode = (
  type: FlowNodeType,
  position: GraphPoint,
  values: Record<string, unknown> = {},
): FlowGraphNode => ({
  id: flowGraphId(type),
  type,
  position,
  values: type === 'inputAction' ? { action: 'Jump', ...values } : { ...values },
});

export const createDefaultFlowGraph = (): FlowGraph => ({
  version: 1,
  variables: [],
  nodes: [createFlowNode('beginPlay', [120, 140])],
  connections: [],
  viewport: { x: 40, y: 40, zoom: 1 },
});

export const createFlowAsset = (name = 'NewFlow'): FlowAssetJson => ({
  version: 1,
  assetType: 'flow',
  name,
  graph: createDefaultFlowGraph(),
});

const flowNodeTypes = new Set<FlowNodeType>(Object.keys(flowNodeDefinitions) as FlowNodeType[]);
const flowValueTypes = new Set<FlowValueType>([
  'bool',
  'int',
  'float',
  'vec2',
  'vec3',
  'vec4',
  'string',
  'name',
  'entity',
  'component',
  'any',
]);

const isPoint = (value: unknown): value is GraphPoint =>
  Array.isArray(value) &&
  value.length === 2 &&
  value.every((coordinate) => typeof coordinate === 'number' && Number.isFinite(coordinate));

const isViewport = (value: unknown): value is GraphViewport => {
  if (!value || typeof value !== 'object') return false;
  const viewport = value as Partial<GraphViewport>;
  return (
    typeof viewport.x === 'number' &&
    Number.isFinite(viewport.x) &&
    typeof viewport.y === 'number' &&
    Number.isFinite(viewport.y) &&
    typeof viewport.zoom === 'number' &&
    Number.isFinite(viewport.zoom) &&
    viewport.zoom > 0
  );
};

const isPinRef = (value: unknown) => {
  if (!value || typeof value !== 'object') return false;
  const ref = value as { nodeId?: unknown; pin?: unknown };
  return typeof ref.nodeId === 'string' && typeof ref.pin === 'string';
};

const isVariable = (value: unknown): value is FlowVariableDefinition => {
  if (!value || typeof value !== 'object') return false;
  const variable = value as Partial<FlowVariableDefinition>;
  return (
    typeof variable.id === 'string' &&
    typeof variable.name === 'string' &&
    typeof variable.type === 'string' &&
    flowValueTypes.has(variable.type as FlowValueType) &&
    typeof variable.exposed === 'boolean'
  );
};

export const isFlowGraph = (value: unknown): value is FlowGraph => {
  if (!value || typeof value !== 'object') return false;
  const graph = value as Partial<FlowGraph>;
  if (graph.version !== 1 || !Array.isArray(graph.nodes) || !Array.isArray(graph.connections)) return false;
  if (!Array.isArray(graph.variables) || !graph.variables.every(isVariable) || !isViewport(graph.viewport))
    return false;

  const nodeIds = new Set<string>();
  for (const node of graph.nodes) {
    if (!node || typeof node !== 'object') return false;
    const candidate = node as Partial<FlowGraphNode>;
    if (
      typeof candidate.id !== 'string' ||
      nodeIds.has(candidate.id) ||
      typeof candidate.type !== 'string' ||
      !flowNodeTypes.has(candidate.type as FlowNodeType) ||
      !isPoint(candidate.position) ||
      !candidate.values ||
      typeof candidate.values !== 'object' ||
      Array.isArray(candidate.values)
    )
      return false;
    nodeIds.add(candidate.id);
  }

  return graph.connections.every((connection) => {
    if (!connection || typeof connection !== 'object') return false;
    const candidate = connection as Partial<FlowGraphConnection>;
    return (
      typeof candidate.id === 'string' &&
      (candidate.kind === 'execution' || candidate.kind === 'value') &&
      isPinRef(candidate.from) &&
      isPinRef(candidate.to) &&
      nodeIds.has(candidate.from!.nodeId) &&
      nodeIds.has(candidate.to!.nodeId)
    );
  });
};

export const isFlowAssetJson = (value: unknown): value is FlowAssetJson => {
  if (!value || typeof value !== 'object') return false;
  const asset = value as Partial<FlowAssetJson>;
  return (
    asset.version === 1 && asset.assetType === 'flow' && typeof asset.name === 'string' && isFlowGraph(asset.graph)
  );
};

export const flowGraphFromAsset = (asset: FlowAssetJson): FlowGraph => {
  if (!isFlowAssetJson(asset)) throw new Error('Flow asset does not contain a valid Flow graph');
  return cloneFlowGraph(asset.graph);
};
