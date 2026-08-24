export type MaterialGraphValueType = 'float' | 'vec2' | 'vec3' | 'vec4' | 'texture2d';
export type MaterialGraphPinType = MaterialGraphValueType | 'numeric';

export type MaterialGraphNodeType =
  | 'output'
  | 'constant'
  | 'vector2'
  | 'vector3'
  | 'vector4'
  | 'textureSample'
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
  | 'normalMap';

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

/**
 * Stable ARC-owned graph representation persisted inside `.arcmat` assets.
 * UI implementation details (DOM ids, React state, selection, Dockview state,
 * etc.) must never be serialized into this structure.
 */
export type MaterialGraph = {
  version: 1;
  nodes: MaterialGraphNode[];
  connections: MaterialGraphConnection[];
  viewport?: MaterialGraphViewport;
};

export type MaterialAssetJson = Record<string, unknown> & {
  version?: number;
  name?: string;
  shaderPath?: string;
  domain?: string;
  blendMode?: string;
  shadingModel?: string;
  doubleSided?: boolean;
  graph?: MaterialGraph | null;
};

export type MaterialNodePin = {
  id: string;
  label: string;
  type: MaterialGraphPinType;
};

export type MaterialNodeCategory = 'Output' | 'Values' | 'Textures' | 'Math' | 'Utility';
export type MaterialNodeSubcategory =
  | 'Surface'
  | 'Constants'
  | 'Sampling'
  | 'Arithmetic'
  | 'Rounding'
  | 'Exponential'
  | 'Trigonometry'
  | 'Range & Interpolation'
  | 'Comparison'
  | 'Measurement'
  | 'Coordinates'
  | 'Animation';

export type MaterialNodeDefinition = {
  type: MaterialGraphNodeType;
  title: string;
  category: MaterialNodeCategory;
  subcategory: MaterialNodeSubcategory;
  inputs: MaterialNodePin[];
  outputs: MaterialNodePin[];
  defaultValues: Record<string, unknown>;
};

const pin = (id: string, label: string, type: MaterialGraphPinType): MaterialNodePin => ({ id, label, type });
const numeric = (id: string, label: string) => pin(id, label, 'numeric');

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

export const materialNodeCategoryOrder: MaterialNodeCategory[] = ['Values', 'Textures', 'Math', 'Utility'];

export const materialNodeSubcategoryOrder: Record<
  Exclude<MaterialNodeCategory, 'Output'>,
  MaterialNodeSubcategory[]
> = {
  Values: ['Constants'],
  Textures: ['Sampling'],
  Math: ['Arithmetic', 'Rounding', 'Exponential', 'Trigonometry', 'Range & Interpolation', 'Comparison', 'Measurement'],
  Utility: ['Coordinates', 'Animation'],
};

export const materialNodeDefinitions: Record<MaterialGraphNodeType, MaterialNodeDefinition> = {
  output: {
    type: 'output',
    title: 'Material Output',
    category: 'Output',
    subcategory: 'Surface',
    inputs: [
      pin('baseColor', 'Base Color', 'vec3'),
      pin('metallic', 'Metallic', 'float'),
      pin('roughness', 'Roughness', 'float'),
      pin('normal', 'Normal', 'vec3'),
      pin('clearCoatNormal', 'Clear Coat Normal', 'vec3'),
      pin('tangent', 'Tangent', 'vec3'),
      pin('ao', 'Ambient Occlusion', 'float'),
      pin('emissive', 'Emissive', 'vec3'),
      pin('opacity', 'Opacity', 'float'),
      pin('alphaClip', 'Alpha Clip', 'float'),
      pin('indexOfRefraction', 'Index of Refraction', 'float'),
      pin('clearCoat', 'Clear Coat', 'float'),
      pin('clearCoatRoughness', 'Clear Coat Roughness', 'float'),
      pin('sheen', 'Sheen', 'float'),
      pin('sheenColor', 'Sheen Color', 'vec3'),
      pin('sheenRoughness', 'Sheen Roughness', 'float'),
      pin('anisotropy', 'Anisotropy', 'float'),
      pin('anisotropyRotation', 'Anisotropy Rotation', 'float'),
      pin('transmission', 'Transmission', 'float'),
      pin('thickness', 'Thickness', 'float'),
      pin('attenuationColor', 'Attenuation Color', 'vec3'),
      pin('attenuationDistance', 'Attenuation Distance', 'float'),
      pin('subsurfaceColor', 'Subsurface Color', 'vec3'),
      pin('subsurface', 'Subsurface', 'float'),
    ],
    outputs: [],
    defaultValues: {},
  },
  constant: {
    type: 'constant',
    title: 'Constant',
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
    title: 'Vector 3 / Color',
    category: 'Values',
    subcategory: 'Constants',
    inputs: [],
    outputs: [pin('value', 'Value', 'vec3')],
    defaultValues: { value: [0.78, 0.8, 0.84] },
  },
  vector4: {
    type: 'vector4',
    title: 'Vector 4',
    category: 'Values',
    subcategory: 'Constants',
    inputs: [],
    outputs: [pin('value', 'Value', 'vec4')],
    defaultValues: { value: [1, 1, 1, 1] },
  },
  textureSample: {
    type: 'textureSample',
    title: 'Texture Sample',
    category: 'Textures',
    subcategory: 'Sampling',
    inputs: [pin('uv', 'UV', 'vec2')],
    outputs: [
      pin('rgb', 'RGB', 'vec3'),
      pin('rgba', 'RGBA', 'vec4'),
      pin('r', 'R', 'float'),
      pin('g', 'G', 'float'),
      pin('b', 'B', 'float'),
      pin('a', 'A', 'float'),
    ],
    defaultValues: { texture: '' },
  },
  texCoord: {
    type: 'texCoord',
    title: 'Texture Coordinate',
    category: 'Utility',
    subcategory: 'Coordinates',
    inputs: [],
    outputs: [pin('uv', 'UV', 'vec2')],
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
  normalMap: {
    type: 'normalMap',
    title: 'Normal Map',
    category: 'Textures',
    subcategory: 'Sampling',
    inputs: [pin('texture', 'Texture RGB', 'vec3')],
    outputs: [pin('normal', 'Normal', 'vec3')],
    defaultValues: { strength: 1 },
  },
};

let generatedId = 0;
export const materialGraphId = (prefix: string) =>
  `${prefix}-${Date.now().toString(36)}-${(generatedId++).toString(36)}`;

export const cloneMaterialGraph = (graph: MaterialGraph): MaterialGraph =>
  JSON.parse(JSON.stringify(graph)) as MaterialGraph;

export const createMaterialNode = (
  type: MaterialGraphNodeType,
  position: MaterialGraphPosition,
  values: Record<string, unknown> = {},
): MaterialGraphNode => ({
  id: type === 'output' ? 'material-output' : materialGraphId(type),
  type,
  position,
  values: { ...materialNodeDefinitions[type].defaultValues, ...values },
});

export const createDefaultMaterialGraph = (): MaterialGraph => {
  const baseColor = createMaterialNode('vector3', [80, 120], { value: [0.78, 0.8, 0.84] });
  baseColor.parameter = { exposed: true, name: 'Base Color' };
  const metallic = createMaterialNode('constant', [80, 290], { value: 0 });
  metallic.parameter = { exposed: true, name: 'Metallic' };
  const roughness = createMaterialNode('constant', [80, 420], { value: 0.62 });
  roughness.parameter = { exposed: true, name: 'Roughness' };
  const output = createMaterialNode('output', [520, 210]);

  return {
    version: 1,
    nodes: [baseColor, metallic, roughness, output],
    connections: [
      {
        id: materialGraphId('connection'),
        from: { nodeId: baseColor.id, pin: 'value' },
        to: { nodeId: output.id, pin: 'baseColor' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: metallic.id, pin: 'value' },
        to: { nodeId: output.id, pin: 'metallic' },
      },
      {
        id: materialGraphId('connection'),
        from: { nodeId: roughness.id, pin: 'value' },
        to: { nodeId: output.id, pin: 'roughness' },
      },
    ],
    viewport: { x: 40, y: 40, zoom: 1 },
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
    graph.nodes.every(
      (node) =>
        Boolean(node) &&
        typeof node.id === 'string' &&
        materialNodeTypes.has(node.type) &&
        Array.isArray(node.position) &&
        node.position.length === 2 &&
        node.position.every((coordinate) => typeof coordinate === 'number' && Number.isFinite(coordinate)),
    ) &&
    Array.isArray(graph.connections)
  );
};

export const materialGraphFromAsset = (asset: MaterialAssetJson): MaterialGraph => {
  if (!isMaterialGraph(asset.graph)) throw new Error('Material asset does not contain a valid native material graph');
  return cloneMaterialGraph(asset.graph);
};
