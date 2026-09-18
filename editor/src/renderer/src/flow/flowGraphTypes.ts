import type { GraphConnectionLike, GraphNodeDefinition, GraphNodeLike, GraphPoint, GraphViewport } from '../graph';

export type FlowValueType =
  'bool' | 'int' | 'float' | 'vec2' | 'vec3' | 'vec4' | 'string' | 'name' | 'entity' | 'component' | 'any';

export type FlowPinType =
  | { kind: 'execution' }
  | {
      kind: 'value';
      valueType: FlowValueType;
    };

export type FlowNodeType =
  | 'beginPlay'
  | 'endPlay'
  | 'tick'
  | 'fixedTick'
  | 'inputAction'
  | 'customEvent'
  | 'branch'
  | 'sequence'
  | 'switchInt'
  | 'doOnce'
  | 'gate'
  | 'forLoop'
  | 'whileLoop'
  | 'delay'
  | 'retriggerableDelay'
  | 'timer'
  | 'callCustomEvent'
  | 'graphInput'
  | 'graphOutput'
  | 'selfEntity'
  | 'createEntity'
  | 'destroyEntity'
  | 'hasCoreComponent'
  | 'removeCoreComponent'
  | 'isEntityAlive'
  | 'getName'
  | 'setName'
  | 'getTag'
  | 'setTag'
  | 'getActive'
  | 'setActive'
  | 'getTransform'
  | 'setTransform'
  | 'boolLiteral'
  | 'intLiteral'
  | 'floatLiteral'
  | 'vector2Literal'
  | 'stringLiteral'
  | 'vector3Literal'
  | 'vector4Literal'
  | 'getVariable'
  | 'setVariable'
  | 'add'
  | 'subtract'
  | 'multiply'
  | 'divide'
  | 'compare'
  | 'boolAnd'
  | 'boolOr'
  | 'boolNot'
  | 'vectorDot'
  | 'vectorLength'
  | 'vectorNormalize'
  | 'vectorScale'
  | 'select'
  | 'convertNumber';

export type FlowNodeCategory =
  'Events' | 'Input' | 'Flow Control' | 'Interface' | 'Entity' | 'Components' | 'Values' | 'Variables' | 'Math';
export type FlowNodeSubcategory =
  | 'Lifecycle'
  | 'Update'
  | 'Actions'
  | 'Branching'
  | 'Sequencing'
  | 'Switching'
  | 'Stateful'
  | 'Looping'
  | 'Timing'
  | 'Custom'
  | 'Graph Interface'
  | 'Identity'
  | 'Lifetime'
  | 'State'
  | 'Transform'
  | 'Literals'
  | 'Access'
  | 'Arithmetic'
  | 'Comparison'
  | 'Boolean'
  | 'Vector'
  | 'Selection'
  | 'Conversion';

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

export type FlowInterfaceValueDefinition = {
  id: string;
  name: string;
  type: FlowValueType;
  defaultValue: unknown;
};

export type FlowEventDefinition = {
  id: string;
  name: string;
};

export type FlowGraph = {
  version: 1;
  variables: FlowVariableDefinition[];
  inputs?: FlowInterfaceValueDefinition[];
  outputs?: FlowInterfaceValueDefinition[];
  events?: FlowEventDefinition[];
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

export type FlowNodeDefinition = GraphNodeDefinition<FlowNodeType, FlowPinType, FlowNodeCategory, FlowNodeSubcategory>;

const execution = (id: string, label: string) => ({ id, label, type: { kind: 'execution' } as FlowPinType });
const value = (id: string, label: string, valueType: FlowValueType) => ({
  id,
  label,
  type: { kind: 'value', valueType } as FlowPinType,
});

export const flowNodeDefinitions: Record<FlowNodeType, FlowNodeDefinition> = {
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
  customEvent: {
    type: 'customEvent',
    title: 'Custom Event',
    category: 'Events',
    subcategory: 'Custom',
    inputs: [],
    outputs: [execution('exec', 'Then')],
  },
  branch: {
    type: 'branch',
    title: 'Branch',
    category: 'Flow Control',
    subcategory: 'Branching',
    inputs: [execution('exec', 'In'), value('condition', 'Condition', 'bool')],
    outputs: [execution('true', 'True'), execution('false', 'False')],
  },
  sequence: {
    type: 'sequence',
    title: 'Sequence',
    category: 'Flow Control',
    subcategory: 'Sequencing',
    inputs: [execution('exec', 'In')],
    outputs: [
      execution('then0', 'Then 0'),
      execution('then1', 'Then 1'),
      execution('then2', 'Then 2'),
      execution('then3', 'Then 3'),
    ],
  },
  switchInt: {
    type: 'switchInt',
    title: 'Switch Integer',
    category: 'Flow Control',
    subcategory: 'Switching',
    inputs: [execution('exec', 'In'), value('selection', 'Selection', 'int')],
    outputs: [
      execution('case0', 'Case 0'),
      execution('case1', 'Case 1'),
      execution('case2', 'Case 2'),
      execution('case3', 'Case 3'),
      execution('default', 'Default'),
    ],
  },
  doOnce: {
    type: 'doOnce',
    title: 'Do Once',
    category: 'Flow Control',
    subcategory: 'Stateful',
    inputs: [execution('exec', 'In'), execution('reset', 'Reset')],
    outputs: [execution('then', 'Then')],
  },
  gate: {
    type: 'gate',
    title: 'Gate',
    category: 'Flow Control',
    subcategory: 'Stateful',
    inputs: [
      execution('enter', 'Enter'),
      execution('open', 'Open'),
      execution('close', 'Close'),
      execution('toggle', 'Toggle'),
    ],
    outputs: [execution('exit', 'Exit')],
  },
  forLoop: {
    type: 'forLoop',
    title: 'For Loop',
    category: 'Flow Control',
    subcategory: 'Looping',
    inputs: [execution('exec', 'In'), value('first', 'First', 'int'), value('last', 'Last', 'int')],
    outputs: [execution('loopBody', 'Loop Body'), execution('completed', 'Completed'), value('index', 'Index', 'int')],
  },
  whileLoop: {
    type: 'whileLoop',
    title: 'While Loop',
    category: 'Flow Control',
    subcategory: 'Looping',
    inputs: [execution('exec', 'In'), value('condition', 'Condition', 'bool')],
    outputs: [execution('loopBody', 'Loop Body'), execution('completed', 'Completed')],
  },
  delay: {
    type: 'delay',
    title: 'Delay',
    category: 'Flow Control',
    subcategory: 'Timing',
    inputs: [execution('exec', 'In'), value('duration', 'Duration', 'float')],
    outputs: [execution('completed', 'Completed')],
  },
  retriggerableDelay: {
    type: 'retriggerableDelay',
    title: 'Retriggerable Delay',
    category: 'Flow Control',
    subcategory: 'Timing',
    inputs: [execution('exec', 'In'), value('duration', 'Duration', 'float')],
    outputs: [execution('completed', 'Completed')],
  },
  timer: {
    type: 'timer',
    title: 'Timer',
    category: 'Flow Control',
    subcategory: 'Timing',
    inputs: [
      execution('start', 'Start'),
      execution('stop', 'Stop'),
      value('interval', 'Interval', 'float'),
      value('looping', 'Looping', 'bool'),
    ],
    outputs: [
      execution('started', 'Started'),
      execution('tick', 'Tick'),
      execution('completed', 'Completed'),
      execution('stopped', 'Stopped'),
      value('active', 'Active', 'bool'),
    ],
  },
  callCustomEvent: {
    type: 'callCustomEvent',
    title: 'Call Custom Event',
    category: 'Events',
    subcategory: 'Custom',
    inputs: [execution('exec', 'In')],
    outputs: [execution('then', 'Then')],
  },
  graphInput: {
    type: 'graphInput',
    title: 'Graph Input',
    category: 'Interface',
    subcategory: 'Graph Interface',
    inputs: [],
    outputs: [value('value', 'Value', 'any')],
  },
  graphOutput: {
    type: 'graphOutput',
    title: 'Graph Output',
    category: 'Interface',
    subcategory: 'Graph Interface',
    inputs: [execution('exec', 'In'), value('value', 'Value', 'any')],
    outputs: [execution('then', 'Then')],
  },
  selfEntity: {
    type: 'selfEntity',
    title: 'Self Entity',
    category: 'Entity',
    subcategory: 'Identity',
    inputs: [],
    outputs: [value('entity', 'Entity', 'entity')],
  },
  createEntity: {
    type: 'createEntity',
    title: 'Create Entity',
    category: 'Entity',
    subcategory: 'Lifetime',
    inputs: [execution('exec', 'In')],
    outputs: [execution('then', 'Then'), value('entity', 'Entity', 'entity')],
  },
  destroyEntity: {
    type: 'destroyEntity',
    title: 'Destroy Entity',
    category: 'Entity',
    subcategory: 'Lifetime',
    inputs: [execution('exec', 'In'), value('entity', 'Entity', 'entity')],
    outputs: [execution('then', 'Then')],
  },
  hasCoreComponent: {
    type: 'hasCoreComponent',
    title: 'Has Component',
    category: 'Components',
    subcategory: 'State',
    inputs: [execution('exec', 'In'), value('entity', 'Entity', 'entity')],
    outputs: [execution('then', 'Then'), value('has', 'Has', 'bool')],
  },
  removeCoreComponent: {
    type: 'removeCoreComponent',
    title: 'Remove Component',
    category: 'Components',
    subcategory: 'State',
    inputs: [execution('exec', 'In'), value('entity', 'Entity', 'entity')],
    outputs: [execution('then', 'Then')],
  },
  isEntityAlive: {
    type: 'isEntityAlive',
    title: 'Is Entity Alive',
    category: 'Entity',
    subcategory: 'State',
    inputs: [execution('exec', 'In'), value('entity', 'Entity', 'entity')],
    outputs: [execution('then', 'Then'), value('alive', 'Alive', 'bool')],
  },
  getName: {
    type: 'getName',
    title: 'Get Name',
    category: 'Components',
    subcategory: 'Identity',
    inputs: [execution('exec', 'In'), value('entity', 'Entity', 'entity')],
    outputs: [execution('then', 'Then'), value('name', 'Name', 'string')],
  },
  setName: {
    type: 'setName',
    title: 'Set Name',
    category: 'Components',
    subcategory: 'Identity',
    inputs: [execution('exec', 'In'), value('entity', 'Entity', 'entity'), value('name', 'Name', 'string')],
    outputs: [execution('then', 'Then')],
  },
  getTag: {
    type: 'getTag',
    title: 'Get Tag',
    category: 'Components',
    subcategory: 'Identity',
    inputs: [execution('exec', 'In'), value('entity', 'Entity', 'entity')],
    outputs: [execution('then', 'Then'), value('tag', 'Tag', 'string')],
  },
  setTag: {
    type: 'setTag',
    title: 'Set Tag',
    category: 'Components',
    subcategory: 'Identity',
    inputs: [execution('exec', 'In'), value('entity', 'Entity', 'entity'), value('tag', 'Tag', 'string')],
    outputs: [execution('then', 'Then')],
  },
  getActive: {
    type: 'getActive',
    title: 'Get Active',
    category: 'Components',
    subcategory: 'State',
    inputs: [execution('exec', 'In'), value('entity', 'Entity', 'entity')],
    outputs: [execution('then', 'Then'), value('active', 'Active', 'bool')],
  },
  setActive: {
    type: 'setActive',
    title: 'Set Active',
    category: 'Components',
    subcategory: 'State',
    inputs: [execution('exec', 'In'), value('entity', 'Entity', 'entity'), value('active', 'Active', 'bool')],
    outputs: [execution('then', 'Then')],
  },
  getTransform: {
    type: 'getTransform',
    title: 'Get Transform',
    category: 'Components',
    subcategory: 'Transform',
    inputs: [execution('exec', 'In'), value('entity', 'Entity', 'entity')],
    outputs: [
      execution('then', 'Then'),
      value('position', 'Position', 'vec3'),
      value('rotation', 'Rotation', 'vec4'),
      value('scale', 'Scale', 'vec3'),
    ],
  },
  setTransform: {
    type: 'setTransform',
    title: 'Set Transform',
    category: 'Components',
    subcategory: 'Transform',
    inputs: [
      execution('exec', 'In'),
      value('entity', 'Entity', 'entity'),
      value('position', 'Position', 'vec3'),
      value('rotation', 'Rotation', 'vec4'),
      value('scale', 'Scale', 'vec3'),
    ],
    outputs: [execution('then', 'Then')],
  },
  boolLiteral: {
    type: 'boolLiteral',
    title: 'Boolean',
    category: 'Values',
    subcategory: 'Literals',
    inputs: [],
    outputs: [value('value', 'Value', 'bool')],
  },
  intLiteral: {
    type: 'intLiteral',
    title: 'Integer',
    category: 'Values',
    subcategory: 'Literals',
    inputs: [],
    outputs: [value('value', 'Value', 'int')],
  },
  floatLiteral: {
    type: 'floatLiteral',
    title: 'Float',
    category: 'Values',
    subcategory: 'Literals',
    inputs: [],
    outputs: [value('value', 'Value', 'float')],
  },
  vector2Literal: {
    type: 'vector2Literal',
    title: 'Vector2',
    category: 'Values',
    subcategory: 'Literals',
    inputs: [],
    outputs: [value('value', 'Value', 'vec2')],
  },
  stringLiteral: {
    type: 'stringLiteral',
    title: 'String',
    category: 'Values',
    subcategory: 'Literals',
    inputs: [],
    outputs: [value('value', 'Value', 'string')],
  },
  vector3Literal: {
    type: 'vector3Literal',
    title: 'Vector3',
    category: 'Values',
    subcategory: 'Literals',
    inputs: [],
    outputs: [value('value', 'Value', 'vec3')],
  },
  vector4Literal: {
    type: 'vector4Literal',
    title: 'Vector4',
    category: 'Values',
    subcategory: 'Literals',
    inputs: [],
    outputs: [value('value', 'Value', 'vec4')],
  },
  getVariable: {
    type: 'getVariable',
    title: 'Get Variable',
    category: 'Variables',
    subcategory: 'Access',
    inputs: [],
    outputs: [value('value', 'Value', 'any')],
  },
  setVariable: {
    type: 'setVariable',
    title: 'Set Variable',
    category: 'Variables',
    subcategory: 'Access',
    inputs: [execution('exec', 'In'), value('value', 'Value', 'any')],
    outputs: [execution('then', 'Then')],
  },
  add: {
    type: 'add',
    title: 'Add',
    category: 'Math',
    subcategory: 'Arithmetic',
    inputs: [value('a', 'A', 'any'), value('b', 'B', 'any')],
    outputs: [value('value', 'Value', 'any')],
  },
  subtract: {
    type: 'subtract',
    title: 'Subtract',
    category: 'Math',
    subcategory: 'Arithmetic',
    inputs: [value('a', 'A', 'any'), value('b', 'B', 'any')],
    outputs: [value('value', 'Value', 'any')],
  },
  multiply: {
    type: 'multiply',
    title: 'Multiply',
    category: 'Math',
    subcategory: 'Arithmetic',
    inputs: [value('a', 'A', 'any'), value('b', 'B', 'any')],
    outputs: [value('value', 'Value', 'any')],
  },
  divide: {
    type: 'divide',
    title: 'Divide',
    category: 'Math',
    subcategory: 'Arithmetic',
    inputs: [value('a', 'A', 'any'), value('b', 'B', 'any')],
    outputs: [value('value', 'Value', 'any')],
  },
  compare: {
    type: 'compare',
    title: 'Compare',
    category: 'Math',
    subcategory: 'Comparison',
    inputs: [value('a', 'A', 'any'), value('b', 'B', 'any')],
    outputs: [value('result', 'Result', 'bool')],
  },
  boolAnd: {
    type: 'boolAnd',
    title: 'AND',
    category: 'Math',
    subcategory: 'Boolean',
    inputs: [value('a', 'A', 'bool'), value('b', 'B', 'bool')],
    outputs: [value('result', 'Result', 'bool')],
  },
  boolOr: {
    type: 'boolOr',
    title: 'OR',
    category: 'Math',
    subcategory: 'Boolean',
    inputs: [value('a', 'A', 'bool'), value('b', 'B', 'bool')],
    outputs: [value('result', 'Result', 'bool')],
  },
  boolNot: {
    type: 'boolNot',
    title: 'NOT',
    category: 'Math',
    subcategory: 'Boolean',
    inputs: [value('value', 'Value', 'bool')],
    outputs: [value('result', 'Result', 'bool')],
  },
  vectorDot: {
    type: 'vectorDot',
    title: 'Dot Product',
    category: 'Math',
    subcategory: 'Vector',
    inputs: [value('a', 'A', 'any'), value('b', 'B', 'any')],
    outputs: [value('value', 'Value', 'float')],
  },
  vectorLength: {
    type: 'vectorLength',
    title: 'Vector Length',
    category: 'Math',
    subcategory: 'Vector',
    inputs: [value('value', 'Vector', 'any')],
    outputs: [value('value', 'Length', 'float')],
  },
  vectorNormalize: {
    type: 'vectorNormalize',
    title: 'Normalize',
    category: 'Math',
    subcategory: 'Vector',
    inputs: [value('value', 'Vector', 'any')],
    outputs: [value('value', 'Value', 'any')],
  },
  vectorScale: {
    type: 'vectorScale',
    title: 'Scale Vector',
    category: 'Math',
    subcategory: 'Vector',
    inputs: [value('vector', 'Vector', 'any'), value('scale', 'Scale', 'float')],
    outputs: [value('value', 'Value', 'any')],
  },
  select: {
    type: 'select',
    title: 'Select',
    category: 'Flow Control',
    subcategory: 'Selection',
    inputs: [
      value('condition', 'Condition', 'bool'),
      value('trueValue', 'True', 'any'),
      value('falseValue', 'False', 'any'),
    ],
    outputs: [value('value', 'Value', 'any')],
  },
  convertNumber: {
    type: 'convertNumber',
    title: 'Convert Number',
    category: 'Math',
    subcategory: 'Conversion',
    inputs: [value('value', 'Value', 'any')],
    outputs: [value('value', 'Value', 'any')],
  },
};

const concreteFlowValueTypes = new Set<FlowValueType>([
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
]);

const nodeConfiguredType = (node: FlowGraphNode): FlowValueType => {
  if (node.type === 'getVariable' || node.type === 'setVariable') {
    const type = node.values.variableType;
    return typeof type === 'string' && concreteFlowValueTypes.has(type as FlowValueType)
      ? (type as FlowValueType)
      : 'any';
  }
  if (node.type === 'graphInput' || node.type === 'graphOutput') {
    const type = node.values.interfaceType;
    return typeof type === 'string' && concreteFlowValueTypes.has(type as FlowValueType)
      ? (type as FlowValueType)
      : 'any';
  }
  const type = node.values.valueType;
  return typeof type === 'string' && concreteFlowValueTypes.has(type as FlowValueType)
    ? (type as FlowValueType)
    : 'any';
};

export const resolveFlowNodeDefinition = (node: FlowGraphNode): FlowNodeDefinition => {
  const definition = flowNodeDefinitions[node.type];
  let inputType = nodeConfiguredType(node);
  let outputType = inputType;

  if (node.type === 'convertNumber') {
    if (node.values.conversion === 'floatToInt') {
      inputType = 'float';
      outputType = 'int';
    } else {
      inputType = 'int';
      outputType = 'float';
    }
  }

  const resolvePin = (pin: (typeof definition.inputs)[number], direction: 'input' | 'output') => {
    if (pin.type.kind !== 'value' || pin.type.valueType !== 'any') return pin;
    return {
      ...pin,
      type: { kind: 'value', valueType: direction === 'input' ? inputType : outputType } as FlowPinType,
    };
  };

  const outputs =
    node.type === 'switchInt'
      ? definition.outputs.map((pin, index) => {
          if (index >= 4) return pin;
          const cases = Array.isArray(node.values.cases) ? node.values.cases : [];
          const caseValue = typeof cases[index] === 'number' ? Math.trunc(cases[index] as number) : index;
          return { ...pin, label: `Case ${caseValue}` };
        })
      : definition.outputs;

  return {
    ...definition,
    inputs: definition.inputs.map((pin) => resolvePin(pin, 'input')),
    outputs: outputs.map((pin) => resolvePin(pin, 'output')),
  };
};

let generatedId = 0;
export const flowGraphId = (prefix: string) => `${prefix}-${Date.now().toString(36)}-${(generatedId++).toString(36)}`;

export const cloneFlowGraph = (graph: FlowGraph): FlowGraph => JSON.parse(JSON.stringify(graph)) as FlowGraph;

const defaultNodeValues = (type: FlowNodeType): Record<string, unknown> => {
  switch (type) {
    case 'inputAction':
      return { action: 'Jump' };
    case 'customEvent':
    case 'callCustomEvent':
      return { eventId: '' };
    case 'graphInput':
    case 'graphOutput':
      return { interfaceId: '', interfaceType: 'float' };
    case 'hasCoreComponent':
    case 'removeCoreComponent':
      return { component: 'transform' };
    case 'boolLiteral':
      return { value: false };
    case 'intLiteral':
    case 'floatLiteral':
      return { value: 0 };
    case 'vector2Literal':
      return { value: [0, 0] };
    case 'stringLiteral':
      return { value: '' };
    case 'vector3Literal':
      return { value: [0, 0, 0] };
    case 'vector4Literal':
      return { value: [0, 0, 0, 1] };
    case 'getVariable':
    case 'setVariable':
      return { variableId: '', variableType: 'float' };
    case 'add':
    case 'subtract':
    case 'multiply':
    case 'divide':
      return { valueType: 'float' };
    case 'compare':
      return { valueType: 'float', operator: 'equal' };
    case 'switchInt':
      return { cases: [0, 1, 2, 3] };
    case 'gate':
      return { startClosed: false };
    case 'vectorDot':
    case 'vectorLength':
    case 'vectorNormalize':
    case 'vectorScale':
      return { valueType: 'vec3' };
    case 'select':
      return { valueType: 'float' };
    case 'convertNumber':
      return { conversion: 'intToFloat' };
    default:
      return {};
  }
};

export const createFlowNode = (
  type: FlowNodeType,
  position: GraphPoint,
  values: Record<string, unknown> = {},
): FlowGraphNode => ({
  id: flowGraphId(type),
  type,
  position,
  values: { ...defaultNodeValues(type), ...values },
});

export const createDefaultFlowGraph = (): FlowGraph => ({
  version: 1,
  variables: [],
  inputs: [],
  outputs: [],
  events: [],
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

const isInterfaceValue = (value: unknown): value is FlowInterfaceValueDefinition => {
  if (!value || typeof value !== 'object') return false;
  const item = value as Partial<FlowInterfaceValueDefinition>;
  return (
    typeof item.id === 'string' &&
    typeof item.name === 'string' &&
    typeof item.type === 'string' &&
    item.type !== 'any' &&
    flowValueTypes.has(item.type as FlowValueType) &&
    'defaultValue' in item
  );
};

const isEvent = (value: unknown): value is FlowEventDefinition => {
  if (!value || typeof value !== 'object') return false;
  const event = value as Partial<FlowEventDefinition>;
  return typeof event.id === 'string' && typeof event.name === 'string';
};

export const isFlowGraph = (value: unknown): value is FlowGraph => {
  if (!value || typeof value !== 'object') return false;
  const graph = value as Partial<FlowGraph>;
  if (graph.version !== 1 || !Array.isArray(graph.nodes) || !Array.isArray(graph.connections)) return false;
  if (!Array.isArray(graph.variables) || !graph.variables.every(isVariable) || !isViewport(graph.viewport))
    return false;
  if (graph.inputs !== undefined && (!Array.isArray(graph.inputs) || !graph.inputs.every(isInterfaceValue))) return false;
  if (graph.outputs !== undefined && (!Array.isArray(graph.outputs) || !graph.outputs.every(isInterfaceValue)))
    return false;
  if (graph.events !== undefined && (!Array.isArray(graph.events) || !graph.events.every(isEvent))) return false;

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
  const graph = cloneFlowGraph(asset.graph);
  graph.inputs ??= [];
  graph.outputs ??= [];
  graph.events ??= [];
  return graph;
};
