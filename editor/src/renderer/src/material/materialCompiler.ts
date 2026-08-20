import {
  materialNodeDefinitions,
  type MaterialAssetSurface,
  type MaterialAssetTextures,
  type MaterialGraph,
  type MaterialGraphConnection,
  type MaterialGraphNode,
  type MaterialGraphValueType,
} from './materialGraphTypes';

export type MaterialCompileDiagnostic = {
  severity: 'warning' | 'error';
  nodeId?: string;
  message: string;
};

export type MaterialIRExpression = {
  id: string;
  operation: MaterialGraphNode['type'];
  resultType: MaterialGraphValueType;
  inputs: string[];
  constant?: number[];
  texture?: string;
};

export type MaterialIR = {
  version: 1;
  expressions: MaterialIRExpression[];
  outputs: Partial<Record<MaterialOutputName, string>>;
  parameters: Array<{
    nodeId: string;
    name: string;
    type: MaterialGraphValueType;
    value: number[];
  }>;
};

export type MaterialOutputName =
  'baseColor' | 'metallic' | 'roughness' | 'normal' | 'ao' | 'emissive' | 'opacity' | 'alphaClip';

export type MaterialCompileResult = {
  succeeded: boolean;
  diagnostics: MaterialCompileDiagnostic[];
  ir: MaterialIR;
  surface: MaterialAssetSurface;
  textures: MaterialAssetTextures;
};

type EvaluatedValue = {
  type: MaterialGraphValueType;
  value?: number[];
  texture?: string;
  textureChannel?: string;
};

const outputNames = new Set<MaterialOutputName>([
  'baseColor',
  'metallic',
  'roughness',
  'normal',
  'ao',
  'emissive',
  'opacity',
  'alphaClip',
]);

const toNumber = (value: unknown, fallback = 0) =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

const toVector = (value: unknown, size: number, fallback = 0): number[] => {
  const source = Array.isArray(value) ? value : [];
  return Array.from({ length: size }, (_, index) => toNumber(source[index], fallback));
};

const pinType = (node: MaterialGraphNode, pin: string, output: boolean): MaterialGraphValueType => {
  const definition = materialNodeDefinitions[node.type];
  return (output ? definition.outputs : definition.inputs).find((candidate) => candidate.id === pin)?.type ?? 'float';
};

const width = (type: MaterialGraphValueType) => {
  switch (type) {
    case 'vec2':
      return 2;
    case 'vec3':
      return 3;
    case 'vec4':
      return 4;
    default:
      return 1;
  }
};

const broadcast = (value: number[], size: number) => {
  if (value.length === size) return [...value];
  if (value.length === 1) return Array.from({ length: size }, () => value[0]);
  return Array.from({ length: size }, (_, index) => value[Math.min(index, value.length - 1)] ?? 0);
};

const resultType = (left: EvaluatedValue | undefined, right?: EvaluatedValue): MaterialGraphValueType => {
  const leftWidth = left ? width(left.type) : 1;
  const rightWidth = right ? width(right.type) : 1;
  const resultWidth = Math.max(leftWidth, rightWidth);
  return resultWidth === 4 ? 'vec4' : resultWidth === 3 ? 'vec3' : resultWidth === 2 ? 'vec2' : 'float';
};

const incomingConnection = (graph: MaterialGraph, nodeId: string, pin: string): MaterialGraphConnection | undefined =>
  graph.connections.find((connection) => connection.to.nodeId === nodeId && connection.to.pin === pin);

const clampNumber = (value: number, minimum: number, maximum: number) => Math.min(maximum, Math.max(minimum, value));

export const compileMaterialGraph = (graph: MaterialGraph): MaterialCompileResult => {
  const diagnostics: MaterialCompileDiagnostic[] = [];
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const expressions = new Map<string, MaterialIRExpression>();
  const evaluated = new Map<string, EvaluatedValue>();
  const visiting = new Set<string>();

  const expressionKey = (nodeId: string, pin: string) => `${nodeId}:${pin}`;

  const evaluateOutput = (nodeId: string, pin: string): EvaluatedValue | undefined => {
    const key = expressionKey(nodeId, pin);
    if (evaluated.has(key)) return evaluated.get(key);
    if (visiting.has(key)) {
      diagnostics.push({ severity: 'error', nodeId, message: 'Material graph contains a cycle.' });
      return undefined;
    }

    const node = nodes.get(nodeId);
    if (!node) {
      diagnostics.push({ severity: 'error', nodeId, message: `Connection references missing node '${nodeId}'.` });
      return undefined;
    }
    visiting.add(key);

    const input = (inputPin: string): EvaluatedValue | undefined => {
      const connection = incomingConnection(graph, node.id, inputPin);
      if (!connection) {
        const defaultValue = node.values[inputPin];
        if (typeof defaultValue === 'number') return { type: 'float', value: [defaultValue] };
        if (Array.isArray(defaultValue)) {
          const size = Math.max(1, Math.min(4, defaultValue.length));
          const type = size === 4 ? 'vec4' : size === 3 ? 'vec3' : size === 2 ? 'vec2' : 'float';
          return { type, value: toVector(defaultValue, size) };
        }
        return undefined;
      }
      return evaluateOutput(connection.from.nodeId, connection.from.pin);
    };

    let value: EvaluatedValue | undefined;
    const expression: MaterialIRExpression = {
      id: key,
      operation: node.type,
      resultType: pinType(node, pin, true),
      inputs: [],
    };

    if (node.type === 'constant') value = { type: 'float', value: [toNumber(node.values.value, 0.5)] };
    else if (node.type === 'vector2') value = { type: 'vec2', value: toVector(node.values.value, 2) };
    else if (node.type === 'vector3') value = { type: 'vec3', value: toVector(node.values.value, 3) };
    else if (node.type === 'vector4') value = { type: 'vec4', value: toVector(node.values.value, 4, 1) };
    else if (node.type === 'textureSample') {
      const texture = typeof node.values.texture === 'string' ? node.values.texture : '';
      value = {
        type: pin === 'rgba' ? 'vec4' : pin === 'rgb' ? 'vec3' : 'float',
        texture,
        textureChannel: pin,
      };
      expression.texture = texture;
      if (!texture)
        diagnostics.push({ severity: 'warning', nodeId: node.id, message: 'Texture Sample has no texture assigned.' });
    } else if (node.type === 'texCoord') {
      value = { type: 'vec2' };
      diagnostics.push({
        severity: 'warning',
        nodeId: node.id,
        message:
          'Texture-coordinate expressions are preserved in Material IR but are not lowered by the descriptor backend yet.',
      });
    } else if (node.type === 'time') {
      value = { type: 'float' };
      diagnostics.push({
        severity: 'warning',
        nodeId: node.id,
        message: 'Time expressions are preserved in Material IR but are not lowered by the descriptor backend yet.',
      });
    } else if (node.type === 'normalMap') {
      const source = input('texture');
      value = source
        ? { type: 'vec3', value: source.value ? broadcast(source.value, 3) : undefined, texture: source.texture }
        : { type: 'vec3' };
      const connection = incomingConnection(graph, node.id, 'texture');
      if (connection) expression.inputs.push(expressionKey(connection.from.nodeId, connection.from.pin));
    } else if (node.type === 'saturate') {
      const source = input('value');
      if (source) {
        value = {
          type: source.type,
          value: source.value?.map((component) => clampNumber(component, 0, 1)),
          texture: source.texture,
        };
      }
      const connection = incomingConnection(graph, node.id, 'value');
      if (connection) expression.inputs.push(expressionKey(connection.from.nodeId, connection.from.pin));
    } else if (node.type === 'clamp') {
      const source = input('value');
      const minimum = input('min')?.value?.[0] ?? toNumber(node.values.min, 0);
      const maximum = input('max')?.value?.[0] ?? toNumber(node.values.max, 1);
      if (source) {
        value = {
          type: source.type,
          value: source.value?.map((component) => clampNumber(component, minimum, maximum)),
          texture: source.texture,
        };
      }
      for (const inputPin of ['value', 'min', 'max']) {
        const connection = incomingConnection(graph, node.id, inputPin);
        if (connection) expression.inputs.push(expressionKey(connection.from.nodeId, connection.from.pin));
      }
    } else if (node.type === 'lerp') {
      const a = input('a');
      const b = input('b');
      const t = input('t');
      const type = resultType(a, b);
      if (a?.value && b?.value && t?.value) {
        const size = width(type);
        const av = broadcast(a.value, size);
        const bv = broadcast(b.value, size);
        const alpha = t.value[0];
        value = { type, value: av.map((component, index) => component + (bv[index] - component) * alpha) };
      } else value = { type };
      for (const inputPin of ['a', 'b', 't']) {
        const connection = incomingConnection(graph, node.id, inputPin);
        if (connection) expression.inputs.push(expressionKey(connection.from.nodeId, connection.from.pin));
      }
    } else if (node.type === 'add' || node.type === 'subtract' || node.type === 'multiply' || node.type === 'divide') {
      const a = input('a');
      const b = input('b');
      const type = resultType(a, b);
      if (a?.value && b?.value) {
        const size = width(type);
        const av = broadcast(a.value, size);
        const bv = broadcast(b.value, size);
        value = {
          type,
          value: av.map((component, index) => {
            if (node.type === 'add') return component + bv[index];
            if (node.type === 'subtract') return component - bv[index];
            if (node.type === 'multiply') return component * bv[index];
            return Math.abs(bv[index]) < 1e-6 ? 0 : component / bv[index];
          }),
        };
      } else value = { type, texture: a?.texture ?? b?.texture };
      for (const inputPin of ['a', 'b']) {
        const connection = incomingConnection(graph, node.id, inputPin);
        if (connection) expression.inputs.push(expressionKey(connection.from.nodeId, connection.from.pin));
      }
    }

    if (value) {
      expression.resultType = value.type;
      if (value.value) expression.constant = [...value.value];
      expressions.set(key, expression);
      evaluated.set(key, value);
    }
    visiting.delete(key);
    return value;
  };

  const output = graph.nodes.find((node) => node.type === 'output');
  if (!output) diagnostics.push({ severity: 'error', message: 'Material graph requires one Material Output node.' });

  const irOutputs: MaterialIR['outputs'] = {};
  const resolvedOutputs = new Map<MaterialOutputName, EvaluatedValue>();
  if (output) {
    for (const inputPin of materialNodeDefinitions.output.inputs) {
      if (!outputNames.has(inputPin.id as MaterialOutputName)) continue;
      const outputName = inputPin.id as MaterialOutputName;
      const connection = incomingConnection(graph, output.id, inputPin.id);
      if (!connection) continue;
      const value = evaluateOutput(connection.from.nodeId, connection.from.pin);
      if (!value) continue;
      irOutputs[outputName] = expressionKey(connection.from.nodeId, connection.from.pin);
      resolvedOutputs.set(outputName, value);
    }
  }

  const baseColor = broadcast(resolvedOutputs.get('baseColor')?.value ?? [0.78, 0.8, 0.84], 3);
  const emissive = broadcast(resolvedOutputs.get('emissive')?.value ?? [0, 0, 0], 3);
  const metallic = clampNumber(resolvedOutputs.get('metallic')?.value?.[0] ?? 0, 0, 1);
  const roughness = clampNumber(resolvedOutputs.get('roughness')?.value?.[0] ?? 0.62, 0, 1);
  const ao = clampNumber(resolvedOutputs.get('ao')?.value?.[0] ?? 1, 0, 1);
  const opacity = clampNumber(resolvedOutputs.get('opacity')?.value?.[0] ?? 1, 0, 1);
  const alphaClip = clampNumber(resolvedOutputs.get('alphaClip')?.value?.[0] ?? 0.5, 0, 1);

  const surface: MaterialAssetSurface = {
    baseColor: { r: baseColor[0], g: baseColor[1], b: baseColor[2], a: opacity },
    metallic,
    roughness,
    normalScale: 1,
    aoStrength: ao,
    emissive: { r: emissive[0], g: emissive[1], b: emissive[2] },
    emissiveStrength: Math.max(emissive[0], emissive[1], emissive[2]) > 0 ? 1 : 0,
    emissiveLuminanceNits: 100,
    alphaCutoff: alphaClip,
  };

  const textures: MaterialAssetTextures = {};
  const baseColorTexture = resolvedOutputs.get('baseColor')?.texture;
  if (baseColorTexture) {
    textures.baseColor = baseColorTexture;
    surface.baseColor = { r: 1, g: 1, b: 1, a: opacity };
  }
  const normalTexture = resolvedOutputs.get('normal')?.texture;
  if (normalTexture) textures.normal = normalTexture;
  const emissiveTexture = resolvedOutputs.get('emissive')?.texture;
  if (emissiveTexture) textures.emissive = emissiveTexture;
  const aoTexture = resolvedOutputs.get('ao')?.texture;
  if (aoTexture) textures.ao = aoTexture;

  const parameters: MaterialIR['parameters'] = [];
  for (const node of graph.nodes) {
    if (!node.parameter?.exposed || node.type === 'output') continue;
    const definition = materialNodeDefinitions[node.type];
    const firstOutput = definition.outputs[0];
    if (!firstOutput) continue;
    const value = evaluateOutput(node.id, firstOutput.id)?.value;
    if (!value) continue;
    parameters.push({
      nodeId: node.id,
      name: node.parameter.name.trim() || definition.title,
      type: firstOutput.type,
      value: [...value],
    });
  }

  const ir: MaterialIR = {
    version: 1,
    expressions: [...expressions.values()],
    outputs: irOutputs,
    parameters,
  };

  return {
    succeeded: diagnostics.every((diagnostic) => diagnostic.severity !== 'error'),
    diagnostics,
    ir,
    surface,
    textures,
  };
};
