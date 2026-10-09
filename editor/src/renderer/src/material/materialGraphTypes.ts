export type MaterialGraphValueType = 'float' | 'vec2' | 'vec3' | 'vec4' | 'texture2d';
export type MaterialTextureDimension = '2d' | 'cube' | '3d';
export type MaterialGraphPinType = MaterialGraphValueType | 'numeric';

export type MaterialDomain = 'surface' | 'terrain';
export type MaterialBlendMode = 'opaque' | 'masked' | 'blend';
export type MaterialShadingModel = 'standard' | 'skin' | 'transmission' | 'unlit' | 'customLit';

export type MaterialSettings = {
  domain: MaterialDomain;
  blendMode: MaterialBlendMode;
  shadingModel: MaterialShadingModel;
  doubleSided: boolean;
  castShadows: boolean;
};

export type MaterialGraphNodeType =
  | 'output'
  | 'constant'
  | 'vector2'
  | 'vector3'
  | 'vector4'
  | 'colorRgba'
  | 'textureSample'
  | 'textureSample2D'
  | 'textureSampleCube'
  | 'textureSample3D'
  | 'worldPosition'
  | 'worldNormal'
  | 'vertexColor'
  | 'texCoord'
  | 'time'
  | 'add'
  | 'subtract'
  | 'multiply'
  | 'divide'
  | 'abs'
  | 'ceil'
  | 'floor'
  | 'round'
  | 'truncate'
  | 'frac'
  | 'fmod'
  | 'min'
  | 'max'
  | 'lerp'
  | 'clamp'
  | 'saturate'
  | 'oneMinus'
  | 'power'
  | 'squareRoot'
  | 'logarithm'
  | 'log2'
  | 'log10'
  | 'sine'
  | 'cosine'
  | 'arcsine'
  | 'arccosine'
  | 'arctangent'
  | 'arctangent2'
  | 'smoothStep'
  | 'step'
  | 'if'
  | 'sign'
  | 'distance'
  | 'length'
  | 'dot'
  | 'normalMap'
  | 'functionInput'
  | 'functionOutput'
  | 'functionCall'
  | 'functionSlot';

export type MaterialGraphPosition = [number, number];

export type MaterialGraphParameter = {
  exposed: boolean;
  name: string;
};

export type MaterialGraphNode = {
  id: string;
  type: MaterialGraphNodeType;
  position: MaterialGraphPosition;
  values: Record<string, unknown>;
  parameter?: MaterialGraphParameter;
};

export type MaterialGraphPinRef = {
  nodeId: string;
  pin: string;
};

export type MaterialGraphConnection = {
  id: string;
  from: MaterialGraphPinRef;
  to: MaterialGraphPinRef;
};

export type MaterialGraphViewport = {
  x: number;
  y: number;
  zoom: number;
};

export type MaterialGraph = {
  version: 1;
  nodes: MaterialGraphNode[];
  connections: MaterialGraphConnection[];
  viewport?: MaterialGraphViewport;
};

export type MaterialFunctionPin = {
  id: string;
  name: string;
  type: Exclude<MaterialGraphValueType, 'texture2d'>;
  default?: number | number[];
};

export type MaterialFunctionAssetJson = {
  kind: 'materialFunction';
  version: 1;
  name: string;
  description?: string;
  inputs: MaterialFunctionPin[];
  outputs: MaterialFunctionPin[];
  graph: MaterialGraph;
};

export type MaterialAssetJson = Record<string, unknown> & {
  version?: number;
  name?: string;
  shaderPath?: string;
  domain?: string;
  blendMode?: string;
  shadingModel?: string;
  doubleSided?: boolean;
  castShadows?: boolean;
  graph?: MaterialGraph | null;
};

export type MaterialNodePin = {
  id: string;
  label: string;
  type: MaterialGraphPinType;
  semanticRange?: MaterialScalarRange;
};

export type MaterialNodeCategory = 'Output' | 'Values' | 'Textures' | 'Math' | 'Utility' | 'Functions';
export type MaterialNodeSubcategory =
  | 'Surface'
  | 'Constants'
  | 'Colors'
  | 'Sampling'
  | 'Arithmetic'
  | 'Rounding'
  | 'Exponential'
  | 'Trigonometry'
  | 'Range & Interpolation'
  | 'Comparison'
  | 'Measurement'
  | 'Coordinates'
  | 'Animation'
  | 'Composition'
  | 'Boundary';

export type MaterialNodeDefinition = {
  type: MaterialGraphNodeType;
  title: string;
  category: MaterialNodeCategory;
  subcategory: MaterialNodeSubcategory;
  inputs: MaterialNodePin[];
  outputs: MaterialNodePin[];
  defaultValues: Record<string, unknown>;
};

const pin = (
  id: string,
  label: string,
  type: MaterialGraphPinType,
  semanticRange?: MaterialScalarRange,
): MaterialNodePin => ({ id, label, type, ...(semanticRange ? { semanticRange } : {}) });
const numeric = (id: string, label: string) => pin(id, label, 'numeric');
const normalized = (id: string, label: string) => pin(id, label, 'float', { min: 0, max: 1 });
const colorOutputs = (): MaterialNodePin[] => [
  pin('rgb', 'RGB', 'vec3'),
  pin('r', 'R', 'float'),
  pin('g', 'G', 'float'),
  pin('b', 'B', 'float'),
  pin('a', 'A', 'float'),
  pin('rgba', 'RGBA', 'vec4'),
];

const unaryMath = (
  type: MaterialGraphNodeType,
  title: string,
  subcategory: MaterialNodeSubcategory,
): MaterialNodeDefinition => ({
  type,
  title,
  category: 'Math',
  subcategory,
  inputs: [numeric('value', 'Value')],
  outputs: [numeric('result', 'Result')],
  defaultValues: {},
});

const binaryMath = (
  type: MaterialGraphNodeType,
  title: string,
  subcategory: MaterialNodeSubcategory,
): MaterialNodeDefinition => ({
  type,
  title,
  category: 'Math',
  subcategory,
  inputs: [numeric('a', 'A'), numeric('b', 'B')],
  outputs: [numeric('result', 'Result')],
  defaultValues: {},
});

export const materialNodeCategoryOrder: MaterialNodeCategory[] = ['Values', 'Textures', 'Functions', 'Math', 'Utility'];

export const materialNodeSubcategoryOrder: Record<
  Exclude<MaterialNodeCategory, 'Output'>,
  MaterialNodeSubcategory[]
> = {
  Values: ['Constants', 'Colors'],
  Textures: ['Sampling'],
  Math: ['Arithmetic', 'Rounding', 'Exponential', 'Trigonometry', 'Range & Interpolation', 'Comparison', 'Measurement'],
  Utility: ['Coordinates', 'Animation'],
  Functions: ['Composition', 'Boundary'],
};

export const materialNodeDefinitions: Record<MaterialGraphNodeType, MaterialNodeDefinition> = {
  output: {
    type: 'output',
    title: 'Material Output',
    category: 'Output',
    subcategory: 'Surface',
    inputs: [
      pin('baseColor', 'Base Color', 'vec3'),
      normalized('metallic', 'Metallic'),
      normalized('roughness', 'Roughness'),
      pin('normal', 'Normal', 'vec3'),
      pin('clearCoatNormal', 'Clear Coat Normal', 'vec3'),
      pin('tangent', 'Tangent', 'vec3'),
      normalized('ao', 'Ambient Occlusion'),
      pin('emissive', 'Emissive', 'vec3'),
      normalized('opacity', 'Opacity'),
      normalized('alphaClip', 'Alpha Clip'),
      pin('indexOfRefraction', 'Index of Refraction', 'float'),
      normalized('clearCoat', 'Clear Coat'),
      normalized('clearCoatRoughness', 'Clear Coat Roughness'),
      normalized('sheen', 'Sheen'),
      pin('sheenColor', 'Sheen Color', 'vec3'),
      normalized('sheenRoughness', 'Sheen Roughness'),
      pin('anisotropy', 'Anisotropy', 'float'),
      pin('anisotropyRotation', 'Anisotropy Rotation', 'float'),
      normalized('transmission', 'Transmission'),
      pin('thickness', 'Thickness', 'float'),
      pin('attenuationColor', 'Attenuation Color', 'vec3'),
      pin('attenuationDistance', 'Attenuation Distance', 'float'),
      pin('subsurfaceColor', 'Subsurface Color', 'vec3'),
      normalized('subsurface', 'Subsurface'),
    ],
    outputs: [],
    defaultValues: {},
  },
  constant: {
    type: 'constant',
    title: 'Scalar',
    category: 'Values',
    subcategory: 'Constants',
    inputs: [],
    outputs: [pin('value', 'Value', 'float')],
    defaultValues: { value: 0.5 },
  },
  vector2: {
    type: 'vector2',
    title: 'Vector 2',
    category: 'Values',
    subcategory: 'Constants',
    inputs: [],
    outputs: [pin('value', 'Value', 'vec2')],
    defaultValues: { value: [0, 0] },
  },
  vector3: {
    type: 'vector3',
    title: 'Vector 3',
    category: 'Values',
    subcategory: 'Constants',
    inputs: [],
    outputs: [pin('value', 'Value', 'vec3')],
    defaultValues: { value: [0, 0, 0] },
  },
  vector4: {
    type: 'vector4',
    title: 'Vector 4',
    category: 'Values',
    subcategory: 'Constants',
    inputs: [],
    outputs: [pin('value', 'Value', 'vec4')],
    defaultValues: { value: [0, 0, 0, 0] },
  },
  colorRgba: {
    type: 'colorRgba',
    title: 'Color',
    category: 'Values',
    subcategory: 'Colors',
    inputs: [],
    outputs: colorOutputs(),
    defaultValues: { value: [1, 1, 1, 1] },
  },
  textureSample: {
    type: 'textureSample',
    title: 'Texture Sample',
    category: 'Textures',
    subcategory: 'Sampling',
    inputs: [pin('uv', 'UV', 'vec2')],
    outputs: colorOutputs(),
    defaultValues: { texture: '', dimension: '2d' },
  },
  textureSample2D: {
    type: 'textureSample2D',
    title: 'Texture Sample 2D',
    category: 'Textures',
    subcategory: 'Sampling',
    inputs: [pin('uv', 'UV', 'vec2')],
    outputs: colorOutputs(),
    defaultValues: { texture: '', dimension: '2d' },
  },
  textureSampleCube: {
    type: 'textureSampleCube',
    title: 'Texture Sample Cube',
    category: 'Textures',
    subcategory: 'Sampling',
    inputs: [pin('uv', 'Direction', 'vec3')],
    outputs: colorOutputs(),
    defaultValues: { texture: '', dimension: 'cube' },
  },
  textureSample3D: {
    type: 'textureSample3D',
    title: 'Texture Sample 3D',
    category: 'Textures',
    subcategory: 'Sampling',
    inputs: [pin('uv', 'UVW', 'vec3')],
    outputs: colorOutputs(),
    defaultValues: { texture: '', dimension: '3d' },
  },
  worldPosition: {
    type: 'worldPosition',
    title: 'World Position',
    category: 'Utility',
    subcategory: 'Coordinates',
    inputs: [],
    outputs: [
      pin('position', 'Position', 'vec3'),
      pin('x', 'X', 'float'),
      pin('y', 'Y', 'float'),
      pin('z', 'Z', 'float'),
    ],
    defaultValues: {},
  },
  worldNormal: {
    type: 'worldNormal',
    title: 'World Normal',
    category: 'Utility',
    subcategory: 'Coordinates',
    inputs: [],
    outputs: [pin('normal', 'Normal', 'vec3'), pin('x', 'X', 'float'), pin('y', 'Y', 'float'), pin('z', 'Z', 'float')],
    defaultValues: {},
  },
  vertexColor: {
    type: 'vertexColor',
    title: 'Vertex Color',
    category: 'Utility',
    subcategory: 'Coordinates',
    inputs: [],
    outputs: colorOutputs(),
    defaultValues: {},
  },
  texCoord: {
    type: 'texCoord',
    title: 'Texture Coordinate',
    category: 'Utility',
    subcategory: 'Coordinates',
    inputs: [],
    outputs: [pin('uv', 'UV0', 'vec2')],
    defaultValues: { channel: 0 },
  },
  time: {
    type: 'time',
    title: 'Time',
    category: 'Utility',
    subcategory: 'Animation',
    inputs: [],
    outputs: [pin('seconds', 'Seconds', 'float')],
    defaultValues: {},
  },
  add: binaryMath('add', 'Add', 'Arithmetic'),
  subtract: binaryMath('subtract', 'Subtract', 'Arithmetic'),
  multiply: binaryMath('multiply', 'Multiply', 'Arithmetic'),
  divide: binaryMath('divide', 'Divide', 'Arithmetic'),
  abs: unaryMath('abs', 'Abs', 'Arithmetic'),
  ceil: unaryMath('ceil', 'Ceil', 'Rounding'),
  floor: unaryMath('floor', 'Floor', 'Rounding'),
  round: unaryMath('round', 'Round', 'Rounding'),
  truncate: unaryMath('truncate', 'Truncate', 'Rounding'),
  frac: unaryMath('frac', 'Frac', 'Rounding'),
  fmod: binaryMath('fmod', 'Fmod / Modulo', 'Arithmetic'),
  min: binaryMath('min', 'Min', 'Arithmetic'),
  max: binaryMath('max', 'Max', 'Arithmetic'),
  lerp: {
    type: 'lerp',
    title: 'Linear Interpolate / Lerp',
    category: 'Math',
    subcategory: 'Range & Interpolation',
    inputs: [numeric('a', 'A'), numeric('b', 'B'), numeric('t', 'Alpha')],
    outputs: [numeric('result', 'Result')],
    defaultValues: {},
  },
  clamp: {
    type: 'clamp',
    title: 'Clamp',
    category: 'Math',
    subcategory: 'Range & Interpolation',
    inputs: [numeric('value', 'Value'), numeric('min', 'Min'), numeric('max', 'Max')],
    outputs: [numeric('result', 'Result')],
    defaultValues: { min: 0, max: 1 },
  },
  saturate: unaryMath('saturate', 'Saturate', 'Range & Interpolation'),
  oneMinus: unaryMath('oneMinus', 'One Minus', 'Arithmetic'),
  power: {
    type: 'power',
    title: 'Power',
    category: 'Math',
    subcategory: 'Exponential',
    inputs: [numeric('base', 'Base'), numeric('exponent', 'Exponent')],
    outputs: [numeric('result', 'Result')],
    defaultValues: {},
  },
  squareRoot: unaryMath('squareRoot', 'Square Root', 'Exponential'),
  logarithm: unaryMath('logarithm', 'Logarithm', 'Exponential'),
  log2: unaryMath('log2', 'Log2', 'Exponential'),
  log10: unaryMath('log10', 'Log10', 'Exponential'),
  sine: unaryMath('sine', 'Sine', 'Trigonometry'),
  cosine: unaryMath('cosine', 'Cosine', 'Trigonometry'),
  arcsine: unaryMath('arcsine', 'Arcsine', 'Trigonometry'),
  arccosine: unaryMath('arccosine', 'Arccosine', 'Trigonometry'),
  arctangent: unaryMath('arctangent', 'Arctangent', 'Trigonometry'),
  arctangent2: {
    type: 'arctangent2',
    title: 'Arctangent2',
    category: 'Math',
    subcategory: 'Trigonometry',
    inputs: [numeric('y', 'Y'), numeric('x', 'X')],
    outputs: [numeric('result', 'Result')],
    defaultValues: {},
  },
  smoothStep: {
    type: 'smoothStep',
    title: 'Smooth Step',
    category: 'Math',
    subcategory: 'Range & Interpolation',
    inputs: [numeric('min', 'Min'), numeric('max', 'Max'), numeric('value', 'Value')],
    outputs: [numeric('result', 'Result')],
    defaultValues: {},
  },
  step: {
    type: 'step',
    title: 'Step',
    category: 'Math',
    subcategory: 'Comparison',
    inputs: [numeric('edge', 'Edge'), numeric('value', 'Value')],
    outputs: [numeric('result', 'Result')],
    defaultValues: {},
  },
  if: {
    type: 'if',
    title: 'If',
    category: 'Math',
    subcategory: 'Comparison',
    inputs: [
      pin('a', 'A', 'float'),
      pin('b', 'B', 'float'),
      numeric('greater', 'A > B'),
      numeric('equal', 'A = B'),
      numeric('less', 'A < B'),
    ],
    outputs: [numeric('result', 'Result')],
    defaultValues: {},
  },
  sign: unaryMath('sign', 'Sign', 'Comparison'),
  distance: {
    ...binaryMath('distance', 'Distance', 'Measurement'),
    outputs: [pin('result', 'Result', 'float')],
  },
  length: {
    ...unaryMath('length', 'Length', 'Measurement'),
    outputs: [pin('result', 'Result', 'float')],
  },
  dot: {
    ...binaryMath('dot', 'Dot Product', 'Measurement'),
    outputs: [pin('result', 'Result', 'float')],
  },
  normalMap: {
    type: 'normalMap',
    title: 'Normal Map',
    category: 'Textures',
    subcategory: 'Sampling',
    inputs: [pin('texture', 'Texture RGB', 'vec3')],
    outputs: [pin('normal', 'Normal', 'vec3')],
    defaultValues: { strength: 1 },
  },
  functionInput: {
    type: 'functionInput',
    title: 'Function Input',
    category: 'Functions',
    subcategory: 'Boundary',
    inputs: [],
    outputs: [pin('value', 'Value', 'float')],
    defaultValues: { input: '', name: 'Input', valueType: 'float' },
  },
  functionOutput: {
    type: 'functionOutput',
    title: 'Function Output',
    category: 'Output',
    subcategory: 'Boundary',
    inputs: [],
    outputs: [],
    defaultValues: { pins: [] },
  },
  functionCall: {
    type: 'functionCall',
    title: 'Material Function',
    category: 'Functions',
    subcategory: 'Composition',
    inputs: [],
    outputs: [],
    defaultValues: { slotId: '', path: '', name: 'Material Function', functions: [], inputPins: [], outputPins: [] },
  },
  functionSlot: {
    type: 'functionSlot',
    title: 'Function Slot',
    category: 'Functions',
    subcategory: 'Composition',
    inputs: [],
    outputs: [],
    defaultValues: { slotId: '', path: '', name: 'Function Slot', inputPins: [], outputPins: [] },
  },
};

let generatedId = 0;
export const materialGraphId = (prefix: string) =>
  `${prefix}-${Date.now().toString(36)}-${(generatedId++).toString(36)}`;

export const cloneMaterialGraph = (graph: MaterialGraph): MaterialGraph =>
  JSON.parse(JSON.stringify(graph)) as MaterialGraph;

export type MaterialScalarRange = { min: number; max: number };

export const materialScalarRange = (node: MaterialGraphNode): MaterialScalarRange | null => {
  if (node.type !== 'constant') return null;
  const min = node.values.min;
  const max = node.values.max;
  if (typeof min !== 'number' || !Number.isFinite(min) || typeof max !== 'number' || !Number.isFinite(max) || min > max)
    return null;
  return { min, max };
};

export const clampMaterialScalarValue = (value: number, range: MaterialScalarRange | null) =>
  range ? Math.min(range.max, Math.max(range.min, value)) : value;

const finiteScalar = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

/**
 * Conservatively infer the scalar domain produced by a material node for authoring feedback.
 *
 * This mirrors the native compiler's safe range inference. Unknown/graph-dependent operations
 * intentionally return null rather than guessing.
 */
export const inferMaterialScalarRange = (
  graph: MaterialGraph,
  nodeId: string,
  visiting: Set<string> = new Set(),
): MaterialScalarRange | null => {
  if (visiting.has(nodeId)) return null;
  const node = graph.nodes.find((candidate) => candidate.id === nodeId);
  if (!node) return null;

  const nextVisiting = new Set(visiting);
  nextVisiting.add(nodeId);
  const inputRange = (pin: string) => {
    const connection = graph.connections.find(
      (candidate) => candidate.to.nodeId === node.id && candidate.to.pin === pin,
    );
    return connection ? inferMaterialScalarRange(graph, connection.from.nodeId, nextVisiting) : null;
  };

  if (node.type === 'constant') {
    const authored = materialScalarRange(node);
    if (authored) return authored;
    const value = finiteScalar(node.values.value);
    return value === null ? null : { min: value, max: value };
  }

  if (node.type === 'saturate') return { min: 0, max: 1 };

  if (node.type === 'clamp') {
    const graphDrivenMin = graph.connections.some(
      (connection) => connection.to.nodeId === node.id && connection.to.pin === 'min',
    );
    const graphDrivenMax = graph.connections.some(
      (connection) => connection.to.nodeId === node.id && connection.to.pin === 'max',
    );
    if (graphDrivenMin || graphDrivenMax) return null;
    const min = finiteScalar(node.values.min);
    const max = finiteScalar(node.values.max);
    return min !== null && max !== null && min <= max ? { min, max } : null;
  }

  if (node.type === 'oneMinus') {
    const value = inputRange('value');
    return value ? { min: 1 - value.max, max: 1 - value.min } : null;
  }

  if (node.type === 'add' || node.type === 'multiply' || node.type === 'min' || node.type === 'max') {
    const a = inputRange('a');
    const b = inputRange('b');
    if (!a || !b) return null;

    if (node.type === 'add') return { min: a.min + b.min, max: a.max + b.max };
    if (node.type === 'multiply') {
      const products = [a.min * b.min, a.min * b.max, a.max * b.min, a.max * b.max];
      return { min: Math.min(...products), max: Math.max(...products) };
    }
    if (node.type === 'min') return { min: Math.min(a.min, b.min), max: Math.min(a.max, b.max) };
    return { min: Math.max(a.min, b.min), max: Math.max(a.max, b.max) };
  }

  return null;
};

export const materialScalarRangeFits = (source: MaterialScalarRange, expected: MaterialScalarRange) =>
  source.min >= expected.min && source.max <= expected.max;

/**
 * Fingerprint only graph data that can change generated material code or runtime bindings.
 * Node positions and the editor viewport are deliberately excluded so graph navigation never
 * recompiles the live material preview.
 */
export const materialGraphCompileFingerprint = (graph: MaterialGraph): string =>
  JSON.stringify({
    version: graph.version,
    nodes: graph.nodes.map((node) => ({
      id: node.id,
      type: node.type,
      values: node.values,
      parameter: node.parameter,
    })),
    connections: graph.connections,
  });

const materialFunctionPins = (value: unknown): MaterialFunctionPin[] =>
  Array.isArray(value)
    ? value.flatMap((candidate) => {
        if (!candidate || typeof candidate !== 'object') return [];
        const pinValue = candidate as Partial<MaterialFunctionPin>;
        if (
          typeof pinValue.id !== 'string' ||
          typeof pinValue.name !== 'string' ||
          (pinValue.type !== 'float' &&
            pinValue.type !== 'vec2' &&
            pinValue.type !== 'vec3' &&
            pinValue.type !== 'vec4')
        )
          return [];
        return [{ id: pinValue.id, name: pinValue.name, type: pinValue.type, default: pinValue.default }];
      })
    : [];

export const createDefaultMaterialFunction = (name: string): MaterialFunctionAssetJson => {
  const input: MaterialFunctionPin = { id: 'value', name: 'Value', type: 'vec3' };
  const output: MaterialFunctionPin = { id: 'result', name: 'Result', type: 'vec3' };
  const inputNode: MaterialGraphNode = {
    id: 'function-input-value',
    type: 'functionInput',
    position: [80, 120],
    values: { input: input.id, name: input.name, valueType: input.type },
  };
  const outputNode: MaterialGraphNode = {
    id: 'function-output',
    type: 'functionOutput',
    position: [520, 120],
    values: { pins: [output] },
  };
  return {
    kind: 'materialFunction',
    version: 1,
    name,
    description: '',
    inputs: [input],
    outputs: [output],
    graph: {
      version: 1,
      nodes: [inputNode, outputNode],
      connections: [
        {
          id: materialGraphId('connection'),
          from: { nodeId: inputNode.id, pin: 'value' },
          to: { nodeId: outputNode.id, pin: output.id },
        },
      ],
      viewport: { x: 40, y: 40, zoom: 1 },
    },
  };
};

export const createMaterialNode = (
  type: MaterialGraphNodeType,
  position: MaterialGraphPosition,
  values: Record<string, unknown> = {},
): MaterialGraphNode => {
  const id = type === 'output' ? 'material-output' : materialGraphId(type);
  return {
    id,
    type,
    position,
    values: {
      ...materialNodeDefinitions[type].defaultValues,
      ...(type === 'functionSlot' || type === 'functionCall' ? { slotId: `slot-${id}` } : {}),
      ...values,
    },
  };
};

export const createDefaultMaterialGraph = (): MaterialGraph => {
  const baseColorTint = createMaterialNode('colorRgba', [80, 80], { value: [0.78, 0.8, 0.84, 1] });
  baseColorTint.parameter = { exposed: true, name: 'Base Color Tint' };
  const baseColorTexture = createMaterialNode('textureSample2D', [80, 240]);
  baseColorTexture.parameter = { exposed: true, name: 'Base Color Texture' };
  const baseColorMultiply = createMaterialNode('multiply', [360, 160]);

  const metallic = createMaterialNode('constant', [80, 440], { value: 0, min: 0, max: 1 });
  metallic.parameter = { exposed: true, name: 'Metallic' };
  const roughness = createMaterialNode('constant', [80, 570], { value: 0.62, min: 0, max: 1 });
  roughness.parameter = { exposed: true, name: 'Roughness' };
  const metallicRoughnessTexture = createMaterialNode('textureSample2D', [80, 700]);
  metallicRoughnessTexture.parameter = { exposed: true, name: 'Metallic Roughness Texture' };
  const metallicMultiply = createMaterialNode('multiply', [360, 460]);
  const roughnessMultiply = createMaterialNode('multiply', [360, 610]);

  const ambientOcclusionTexture = createMaterialNode('textureSample2D', [80, 860]);
  ambientOcclusionTexture.parameter = { exposed: true, name: 'Ambient Occlusion Texture' };

  const normalTexture = createMaterialNode('textureSample2D', [80, 1020], { semantic: 'normal' });
  normalTexture.parameter = { exposed: true, name: 'Normal Texture' };
  const normalMap = createMaterialNode('normalMap', [400, 1020]);

  const clearCoat = createMaterialNode('constant', [80, 1200], { value: 0, min: 0, max: 1 });
  clearCoat.parameter = { exposed: true, name: 'Clear Coat' };
  const clearCoatRoughness = createMaterialNode('constant', [80, 1330], { value: 0.1, min: 0, max: 1 });
  clearCoatRoughness.parameter = { exposed: true, name: 'Clear Coat Roughness' };
  const clearCoatTexture = createMaterialNode('textureSample2D', [80, 1460], { semantic: 'clear_coat' });
  clearCoatTexture.parameter = { exposed: true, name: 'Clear Coat Texture' };
  const clearCoatMultiply = createMaterialNode('multiply', [400, 1210]);
  const clearCoatRoughnessMultiply = createMaterialNode('multiply', [400, 1360]);

  const emissiveColor = createMaterialNode('colorRgba', [80, 1640], { value: [1, 1, 1, 1] });
  emissiveColor.parameter = { exposed: true, name: 'Emissive Color' };
  const emissiveTexture = createMaterialNode('textureSample2D', [80, 1800]);
  emissiveTexture.parameter = { exposed: true, name: 'Emissive Texture' };
  const emissiveStrength = createMaterialNode('constant', [80, 1960], { value: 0 });
  emissiveStrength.parameter = { exposed: true, name: 'Emissive Strength' };
  const emissiveColorMultiply = createMaterialNode('multiply', [400, 1720]);
  const emissiveStrengthMultiply = createMaterialNode('multiply', [600, 1720]);
  const output = createMaterialNode('output', [900, 760]);

  return {
    version: 1,
    nodes: [
      baseColorTint,
      baseColorTexture,
      baseColorMultiply,
      metallic,
      roughness,
      metallicRoughnessTexture,
      metallicMultiply,
      roughnessMultiply,
      ambientOcclusionTexture,
      normalTexture,
      normalMap,
      clearCoat,
      clearCoatRoughness,
      clearCoatTexture,
      clearCoatMultiply,
      clearCoatRoughnessMultiply,
      emissiveColor,
      emissiveTexture,
      emissiveStrength,
      emissiveColorMultiply,
      emissiveStrengthMultiply,
      output,
    ],
    connections: [
      {
        id: materialGraphId('connection'),
        from: { nodeId: baseColorTint.id, pin: 'rgb' },
        to: { nodeId: baseColorMultiply.id, pin: 'a' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: baseColorTexture.id, pin: 'rgb' },
        to: { nodeId: baseColorMultiply.id, pin: 'b' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: baseColorMultiply.id, pin: 'result' },
        to: { nodeId: output.id, pin: 'baseColor' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: metallic.id, pin: 'value' },
        to: { nodeId: metallicMultiply.id, pin: 'a' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: metallicRoughnessTexture.id, pin: 'b' },
        to: { nodeId: metallicMultiply.id, pin: 'b' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: metallicMultiply.id, pin: 'result' },
        to: { nodeId: output.id, pin: 'metallic' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: roughness.id, pin: 'value' },
        to: { nodeId: roughnessMultiply.id, pin: 'a' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: metallicRoughnessTexture.id, pin: 'g' },
        to: { nodeId: roughnessMultiply.id, pin: 'b' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: roughnessMultiply.id, pin: 'result' },
        to: { nodeId: output.id, pin: 'roughness' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: ambientOcclusionTexture.id, pin: 'r' },
        to: { nodeId: output.id, pin: 'ao' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: normalTexture.id, pin: 'rgb' },
        to: { nodeId: normalMap.id, pin: 'texture' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: normalMap.id, pin: 'normal' },
        to: { nodeId: output.id, pin: 'normal' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: clearCoat.id, pin: 'value' },
        to: { nodeId: clearCoatMultiply.id, pin: 'a' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: clearCoatTexture.id, pin: 'r' },
        to: { nodeId: clearCoatMultiply.id, pin: 'b' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: clearCoatMultiply.id, pin: 'result' },
        to: { nodeId: output.id, pin: 'clearCoat' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: clearCoatRoughness.id, pin: 'value' },
        to: { nodeId: clearCoatRoughnessMultiply.id, pin: 'a' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: clearCoatTexture.id, pin: 'g' },
        to: { nodeId: clearCoatRoughnessMultiply.id, pin: 'b' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: clearCoatRoughnessMultiply.id, pin: 'result' },
        to: { nodeId: output.id, pin: 'clearCoatRoughness' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: emissiveColor.id, pin: 'rgb' },
        to: { nodeId: emissiveColorMultiply.id, pin: 'a' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: emissiveTexture.id, pin: 'rgb' },
        to: { nodeId: emissiveColorMultiply.id, pin: 'b' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: emissiveColorMultiply.id, pin: 'result' },
        to: { nodeId: emissiveStrengthMultiply.id, pin: 'a' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: emissiveStrength.id, pin: 'value' },
        to: { nodeId: emissiveStrengthMultiply.id, pin: 'b' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: emissiveStrengthMultiply.id, pin: 'result' },
        to: { nodeId: output.id, pin: 'emissive' },
      },
    ],
    viewport: { x: 40, y: 40, zoom: 0.85 },
  };
};

const materialNodeTypes = new Set<MaterialGraphNodeType>(
  Object.keys(materialNodeDefinitions) as MaterialGraphNodeType[],
);

export const isMaterialGraph = (value: unknown): value is MaterialGraph => {
  if (!value || typeof value !== 'object') return false;
  const graph = value as Partial<MaterialGraph>;
  return (
    graph.version === 1 &&
    Array.isArray(graph.nodes) &&
    graph.nodes.every((node) => {
      if (
        !node ||
        typeof node.id !== 'string' ||
        !materialNodeTypes.has(node.type) ||
        !Array.isArray(node.position) ||
        node.position.length !== 2 ||
        !node.position.every((coordinate) => typeof coordinate === 'number' && Number.isFinite(coordinate))
      )
        return false;

      if (node.type !== 'constant') return true;
      const min = node.values?.min;
      const max = node.values?.max;
      if (min === undefined && max === undefined) return true;
      if (
        typeof min !== 'number' ||
        !Number.isFinite(min) ||
        typeof max !== 'number' ||
        !Number.isFinite(max) ||
        min > max
      )
        return false;
      const scalar = node.values?.value;
      return typeof scalar === 'number' && Number.isFinite(scalar) && scalar >= min && scalar <= max;
    }) &&
    Array.isArray(graph.connections)
  );
};

export const materialGraphFromAsset = (asset: MaterialAssetJson): MaterialGraph => {
  if (!isMaterialGraph(asset.graph)) throw new Error('Material asset does not contain a valid native material graph');
  return cloneMaterialGraph(asset.graph);
};

export const isMaterialTextureSampleNodeType = (type: MaterialGraphNodeType) =>
  type === 'textureSample' || type === 'textureSample2D' || type === 'textureSampleCube' || type === 'textureSample3D';

export const materialTextureDimension = (node: MaterialGraphNode): MaterialTextureDimension => {
  if (node.type === 'textureSampleCube') return 'cube';
  if (node.type === 'textureSample3D') return '3d';
  if (node.type === 'textureSample2D') return '2d';
  const dimension = node.values.dimension;
  return dimension === 'cube' || dimension === '3d' ? dimension : '2d';
};

export const materialNodeDefinition = (node: MaterialGraphNode): MaterialNodeDefinition => {
  const definition = materialNodeDefinitions[node.type];

  if (node.type === 'functionInput') {
    const type =
      node.values.valueType === 'vec2' || node.values.valueType === 'vec3' || node.values.valueType === 'vec4'
        ? node.values.valueType
        : 'float';
    const label = typeof node.values.name === 'string' && node.values.name.trim() ? node.values.name : 'Input';
    return { ...definition, title: label, outputs: [pin('value', label, type)] };
  }

  if (node.type === 'functionOutput') {
    const pins = materialFunctionPins(node.values.pins);
    return { ...definition, inputs: pins.map((value) => pin(value.id, value.name, value.type)) };
  }

  if (node.type === 'functionCall' || node.type === 'functionSlot') {
    const inputs = materialFunctionPins(node.values.inputPins);
    const outputs = materialFunctionPins(node.values.outputPins);
    const title = typeof node.values.name === 'string' && node.values.name.trim() ? node.values.name : definition.title;
    return {
      ...definition,
      title,
      inputs: inputs.map((value) => pin(value.id, value.name, value.type)),
      outputs: outputs.map((value) => pin(value.id, value.name, value.type)),
    };
  }

  if (node.type !== 'textureSample') return definition;

  const dimension = materialTextureDimension(node);
  const coordinate =
    dimension === '2d'
      ? pin('uv', 'UV', 'vec2')
      : dimension === 'cube'
        ? pin('uv', 'Direction', 'vec3')
        : pin('uv', 'UVW', 'vec3');
  return { ...definition, inputs: [coordinate] };
};
