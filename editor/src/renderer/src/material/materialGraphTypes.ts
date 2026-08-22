export type MaterialGraphValueType = 'float' | 'vec2' | 'vec3' | 'vec4' | 'texture2d';

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
  | 'lerp'
  | 'clamp'
  | 'saturate'
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

export type MaterialAssetSurface = {
  baseColor?: { r: number; g: number; b: number; a: number };
  metallic?: number;
  roughness?: number;
  normalScale?: number;
  aoStrength?: number;
  emissive?: { r: number; g: number; b: number };
  emissiveStrength?: number;
  emissiveLuminanceNits?: number;
  alphaCutoff?: number;
};

export type MaterialAssetTextures = {
  baseColor?: string;
  metallicRoughness?: string;
  normal?: string;
  ao?: string;
  emissive?: string;
  height?: string;
  clearCoat?: string;
  clearCoatRoughness?: string;
  clearCoatNormal?: string;
  anisotropy?: string;
  subsurface?: string;
  thickness?: string;
  transmission?: string;
};

export type MaterialAssetJson = Record<string, unknown> & {
  version?: number;
  name?: string;
  shader?: string;
  shaderPath?: string;
  domain?: string;
  blendMode?: string;
  shadingModel?: string;
  doubleSided?: boolean;
  surface?: MaterialAssetSurface;
  textures?: MaterialAssetTextures;
  graph?: MaterialGraph | null;
};

export type MaterialNodePin = {
  id: string;
  label: string;
  type: MaterialGraphValueType;
};

export type MaterialNodeDefinition = {
  type: MaterialGraphNodeType;
  title: string;
  category: 'Output' | 'Values' | 'Textures' | 'Math' | 'Utility';
  inputs: MaterialNodePin[];
  outputs: MaterialNodePin[];
  defaultValues: Record<string, unknown>;
};

const pin = (id: string, label: string, type: MaterialGraphValueType): MaterialNodePin => ({ id, label, type });

export const materialNodeDefinitions: Record<MaterialGraphNodeType, MaterialNodeDefinition> = {
  output: {
    type: 'output',
    title: 'Material Output',
    category: 'Output',
    inputs: [
      pin('baseColor', 'Base Color', 'vec3'),
      pin('metallic', 'Metallic', 'float'),
      pin('roughness', 'Roughness', 'float'),
      pin('normal', 'Normal', 'vec3'),
      pin('ao', 'Ambient Occlusion', 'float'),
      pin('emissive', 'Emissive', 'vec3'),
      pin('opacity', 'Opacity', 'float'),
      pin('alphaClip', 'Alpha Clip', 'float'),
    ],
    outputs: [],
    defaultValues: {},
  },
  constant: {
    type: 'constant',
    title: 'Constant',
    category: 'Values',
    inputs: [],
    outputs: [pin('value', 'Value', 'float')],
    defaultValues: { value: 0.5 },
  },
  vector2: {
    type: 'vector2',
    title: 'Vector 2',
    category: 'Values',
    inputs: [],
    outputs: [pin('value', 'Value', 'vec2')],
    defaultValues: { value: [0, 0] },
  },
  vector3: {
    type: 'vector3',
    title: 'Vector 3 / Color',
    category: 'Values',
    inputs: [],
    outputs: [pin('value', 'Value', 'vec3')],
    defaultValues: { value: [0.78, 0.8, 0.84] },
  },
  vector4: {
    type: 'vector4',
    title: 'Vector 4',
    category: 'Values',
    inputs: [],
    outputs: [pin('value', 'Value', 'vec4')],
    defaultValues: { value: [1, 1, 1, 1] },
  },
  textureSample: {
    type: 'textureSample',
    title: 'Texture Sample',
    category: 'Textures',
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
    inputs: [],
    outputs: [pin('uv', 'UV', 'vec2')],
    defaultValues: { channel: 0 },
  },
  time: {
    type: 'time',
    title: 'Time',
    category: 'Utility',
    inputs: [],
    outputs: [pin('seconds', 'Seconds', 'float')],
    defaultValues: {},
  },
  add: {
    type: 'add',
    title: 'Add',
    category: 'Math',
    inputs: [pin('a', 'A', 'vec4'), pin('b', 'B', 'vec4')],
    outputs: [pin('result', 'Result', 'vec4')],
    defaultValues: {},
  },
  subtract: {
    type: 'subtract',
    title: 'Subtract',
    category: 'Math',
    inputs: [pin('a', 'A', 'vec4'), pin('b', 'B', 'vec4')],
    outputs: [pin('result', 'Result', 'vec4')],
    defaultValues: {},
  },
  multiply: {
    type: 'multiply',
    title: 'Multiply',
    category: 'Math',
    inputs: [pin('a', 'A', 'vec4'), pin('b', 'B', 'vec4')],
    outputs: [pin('result', 'Result', 'vec4')],
    defaultValues: {},
  },
  divide: {
    type: 'divide',
    title: 'Divide',
    category: 'Math',
    inputs: [pin('a', 'A', 'vec4'), pin('b', 'B', 'vec4')],
    outputs: [pin('result', 'Result', 'vec4')],
    defaultValues: {},
  },
  lerp: {
    type: 'lerp',
    title: 'Lerp',
    category: 'Math',
    inputs: [pin('a', 'A', 'vec4'), pin('b', 'B', 'vec4'), pin('t', 'Alpha', 'float')],
    outputs: [pin('result', 'Result', 'vec4')],
    defaultValues: {},
  },
  clamp: {
    type: 'clamp',
    title: 'Clamp',
    category: 'Math',
    inputs: [pin('value', 'Value', 'vec4'), pin('min', 'Min', 'float'), pin('max', 'Max', 'float')],
    outputs: [pin('result', 'Result', 'vec4')],
    defaultValues: { min: 0, max: 1 },
  },
  saturate: {
    type: 'saturate',
    title: 'Saturate',
    category: 'Math',
    inputs: [pin('value', 'Value', 'vec4')],
    outputs: [pin('result', 'Result', 'vec4')],
    defaultValues: {},
  },
  normalMap: {
    type: 'normalMap',
    title: 'Normal Map',
    category: 'Textures',
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

const finite = (value: unknown, fallback: number) =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

const vec3FromAsset = (
  value: MaterialAssetSurface['baseColor'] | MaterialAssetSurface['emissive'],
  fallback: number[],
) => [finite(value?.r, fallback[0]), finite(value?.g, fallback[1]), finite(value?.b, fallback[2])];

export const createDefaultMaterialGraph = (asset: MaterialAssetJson = {}): MaterialGraph => {
  const surface = asset.surface ?? {};
  const baseColor = createMaterialNode('vector3', [80, 120], {
    value: vec3FromAsset(surface.baseColor, [0.78, 0.8, 0.84]),
  });
  baseColor.parameter = { exposed: true, name: 'Base Color' };
  const metallic = createMaterialNode('constant', [80, 290], { value: finite(surface.metallic, 0) });
  metallic.parameter = { exposed: true, name: 'Metallic' };
  const roughness = createMaterialNode('constant', [80, 420], { value: finite(surface.roughness, 0.62) });
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

export const materialGraphFromAsset = (asset: MaterialAssetJson): MaterialGraph =>
  isMaterialGraph(asset.graph) ? cloneMaterialGraph(asset.graph) : createDefaultMaterialGraph(asset);
