import type {
  MaterialAssetSurface,
  MaterialAssetTextures,
  MaterialGraph,
  MaterialGraphConnection,
  MaterialGraphValueType,
} from './materialGraphTypes';

/** Diagnostic returned by ARC's native Material IR/compiler pipeline. */
export type MaterialCompileDiagnostic = {
  severity: 'information' | 'warning' | 'error';
  code?: string;
  nodeId?: string;
  path?: string;
  line?: number;
  column?: number;
  message: string;
};

/** Editor-facing state for the native material compiler. */
export type MaterialCompileResult = {
  status: 'idle' | 'compiling' | 'succeeded' | 'failed';
  succeeded: boolean;
  diagnostics: MaterialCompileDiagnostic[];
};

export type NativeMaterialCompilePayload = {
  succeeded?: boolean;
  message?: string;
  diagnostics?: Array<{
    severity?: string;
    code?: string;
    message?: string;
    path?: string;
    line?: number;
    column?: number;
    graphNode?: string;
  }>;
};

export const emptyMaterialCompileResult = (): MaterialCompileResult => ({
  status: 'idle',
  succeeded: false,
  diagnostics: [],
});

export const compilingMaterialResult = (previous: MaterialCompileResult): MaterialCompileResult => ({
  ...previous,
  status: 'compiling',
});

export const nativeMaterialCompileResult = (
  responseSucceeded: boolean,
  payload: NativeMaterialCompilePayload | undefined,
  fallbackMessage = 'Native material compilation failed',
): MaterialCompileResult => {
  const diagnostics: MaterialCompileDiagnostic[] = (payload?.diagnostics ?? []).map((diagnostic) => ({
    severity:
      diagnostic.severity === 'warning' ? 'warning' : diagnostic.severity === 'information' ? 'information' : 'error',
    code: diagnostic.code,
    nodeId: diagnostic.graphNode || undefined,
    path: diagnostic.path,
    line: diagnostic.line,
    column: diagnostic.column,
    message: diagnostic.message || fallbackMessage,
  }));
  const succeeded = responseSucceeded && payload?.succeeded === true;
  if (!succeeded && diagnostics.length === 0)
    diagnostics.push({ severity: 'error', message: payload?.message || fallbackMessage });
  return { status: succeeded ? 'succeeded' : 'failed', succeeded, diagnostics };
};

export type MaterialEditorParameter = {
  nodeId: string;
  name: string;
  type: MaterialGraphValueType;
};

/**
 * Return authored exposed-parameter metadata for the inspector.
 *
 * This is deliberately not compiler output. Type checking, reachability, parameter IDs, layout,
 * diagnostics, and shader generation are owned exclusively by the native compiler.
 */
export const materialEditorParameters = (graph: MaterialGraph): MaterialEditorParameter[] =>
  graph.nodes.flatMap((node) => {
    if (!node.parameter?.exposed || node.type === 'output' || node.type === 'textureSample') return [];
    const type: MaterialGraphValueType =
      node.type === 'vector2' ? 'vec2' : node.type === 'vector3' ? 'vec3' : node.type === 'vector4' ? 'vec4' : 'float';
    return [{ nodeId: node.id, name: node.parameter.name.trim() || 'Parameter', type }];
  });

/**
 * Compatibility projection used only by the legacy renderer while Stage 11 is not yet merged.
 *
 * This routine never validates a graph and never controls compilation success. It projects simple
 * constant/texture values into legacy `.arcmat` surface fields so editor previews do not regress
 * during the native-compiler migration. It can be deleted with the legacy renderer.
 */
export const projectLegacyMaterialPreview = (
  graph: MaterialGraph,
): { surface: MaterialAssetSurface; textures: MaterialAssetTextures } => {
  type Value = { value?: number[]; texture?: string };
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const visiting = new Set<string>();

  const incoming = (nodeId: string, pin: string): MaterialGraphConnection | undefined =>
    graph.connections.find((connection) => connection.to.nodeId === nodeId && connection.to.pin === pin);

  const vector = (value: unknown, size: number, fallback = 0): number[] => {
    const source = Array.isArray(value) ? value : [];
    return Array.from({ length: size }, (_, index) =>
      typeof source[index] === 'number' && Number.isFinite(source[index]) ? Number(source[index]) : fallback,
    );
  };
  const scalar = (value: unknown, fallback = 0) =>
    typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  const broadcast = (value: number[], size: number) =>
    value.length === size
      ? [...value]
      : Array.from(
          { length: size },
          (_, index) => value[value.length === 1 ? 0 : Math.min(index, value.length - 1)] ?? 0,
        );

  const evaluate = (nodeId: string, pin: string): Value | undefined => {
    const key = `${nodeId}:${pin}`;
    if (visiting.has(key)) return undefined;
    const node = nodes.get(nodeId);
    if (!node) return undefined;
    visiting.add(key);

    const input = (inputPin: string): Value | undefined => {
      const connection = incoming(node.id, inputPin);
      return connection ? evaluate(connection.from.nodeId, connection.from.pin) : undefined;
    };

    let result: Value | undefined;
    if (node.type === 'constant') result = { value: [scalar(node.values.value, 0.5)] };
    else if (node.type === 'vector2') result = { value: vector(node.values.value, 2) };
    else if (node.type === 'vector3') result = { value: vector(node.values.value, 3) };
    else if (node.type === 'vector4') result = { value: vector(node.values.value, 4, 1) };
    else if (node.type === 'textureSample') {
      const texture = typeof node.values.texture === 'string' ? node.values.texture : '';
      result = texture ? { texture } : undefined;
    } else if (node.type === 'normalMap') result = input('texture');
    else if (node.type === 'saturate' || node.type === 'clamp') result = input('value');
    else if (node.type === 'lerp') {
      const a = input('a');
      const b = input('b');
      const t = input('t');
      if (a?.value && b?.value && t?.value) {
        const size = Math.max(a.value.length, b.value.length);
        const av = broadcast(a.value, size);
        const bv = broadcast(b.value, size);
        result = { value: av.map((component, index) => component + (bv[index] - component) * t.value![0]) };
      } else result = { texture: a?.texture ?? b?.texture };
    } else if (['add', 'subtract', 'multiply', 'divide'].includes(node.type)) {
      const a = input('a');
      const b = input('b');
      if (a?.value && b?.value) {
        const size = Math.max(a.value.length, b.value.length);
        const av = broadcast(a.value, size);
        const bv = broadcast(b.value, size);
        result = {
          value: av.map((component, index) => {
            if (node.type === 'add') return component + bv[index];
            if (node.type === 'subtract') return component - bv[index];
            if (node.type === 'multiply') return component * bv[index];
            return Math.abs(bv[index]) < 1e-6 ? 0 : component / bv[index];
          }),
        };
      } else result = { texture: a?.texture ?? b?.texture };
    }

    visiting.delete(key);
    return result;
  };

  const output = graph.nodes.find((node) => node.type === 'output');
  const outputValue = (pin: string): Value | undefined => {
    if (!output) return undefined;
    const connection = incoming(output.id, pin);
    return connection ? evaluate(connection.from.nodeId, connection.from.pin) : undefined;
  };
  const clamp01 = (value: number) => Math.min(1, Math.max(0, value));
  const base = broadcast(outputValue('baseColor')?.value ?? [0.78, 0.8, 0.84], 3);
  const emissive = broadcast(outputValue('emissive')?.value ?? [0, 0, 0], 3);
  const opacity = clamp01(outputValue('opacity')?.value?.[0] ?? 1);
  const surface: MaterialAssetSurface = {
    baseColor: { r: base[0], g: base[1], b: base[2], a: opacity },
    metallic: clamp01(outputValue('metallic')?.value?.[0] ?? 0),
    roughness: clamp01(outputValue('roughness')?.value?.[0] ?? 0.62),
    normalScale: 1,
    aoStrength: clamp01(outputValue('ao')?.value?.[0] ?? 1),
    emissive: { r: emissive[0], g: emissive[1], b: emissive[2] },
    emissiveStrength: Math.max(...emissive) > 0 ? 1 : 0,
    emissiveLuminanceNits: 100,
    alphaCutoff: clamp01(outputValue('alphaClip')?.value?.[0] ?? 0.5),
  };
  const textures: MaterialAssetTextures = {};
  const assignTexture = (pin: string, target: keyof MaterialAssetTextures) => {
    const texture = outputValue(pin)?.texture;
    if (texture) textures[target] = texture;
  };
  assignTexture('baseColor', 'baseColor');
  assignTexture('normal', 'normal');
  assignTexture('ao', 'ao');
  assignTexture('emissive', 'emissive');
  if (textures.baseColor && surface.baseColor) surface.baseColor = { r: 1, g: 1, b: 1, a: opacity };
  return { surface, textures };
};
