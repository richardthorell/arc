import type {
  MaterialGraph,
  MaterialGraphNode,
  MaterialGraphNodeType,
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

export type MaterialGraphEditImpact = 'none' | 'parameter-values' | 'shader';

const sameParameterMetadata = (before: MaterialGraphNode, after: MaterialGraphNode): boolean =>
  JSON.stringify(before.parameter ?? null) === JSON.stringify(after.parameter ?? null);

const parameterNodeTypes = new Set<MaterialGraphNodeType>([
  'constant',
  'vector2',
  'vector3',
  'vector4',
  'colorRgba',
  'textureSample',
  'textureSample2D',
]);

const contributingMaterialNodeIds = (graph: MaterialGraph): Set<string> => {
  const incoming = new Map<string, string[]>();
  for (const connection of graph.connections) {
    const sources = incoming.get(connection.to.nodeId) ?? [];
    sources.push(connection.from.nodeId);
    incoming.set(connection.to.nodeId, sources);
  }

  const contributing = new Set<string>();
  const pending = graph.nodes.filter((node) => node.type === 'output').map((node) => node.id);
  while (pending.length > 0) {
    const nodeId = pending.pop()!;
    if (contributing.has(nodeId)) continue;
    contributing.add(nodeId);
    pending.push(...(incoming.get(nodeId) ?? []));
  }
  return contributing;
};

const effectiveExposedParameterNodeIds = (graph: MaterialGraph): Set<string> => {
  const contributing = contributingMaterialNodeIds(graph);
  return new Set(
    graph.nodes
      .filter(
        (node) => node.parameter?.exposed === true && parameterNodeTypes.has(node.type) && contributing.has(node.id),
      )
      .map((node) => node.id),
  );
};

/**
 * Classify an authored graph edit before deciding whether shader compilation is required.
 *
 * Only value changes on already-exposed, effective parameters are safe to treat as runtime
 * parameter updates. Topology, node identity/type, parameter metadata, and viewport-independent
 * authored structure remain shader-affecting and must go through the authoritative native compiler.
 */
export const materialGraphEditImpact = (before: MaterialGraph, after: MaterialGraph): MaterialGraphEditImpact => {
  if (before === after || JSON.stringify(before) === JSON.stringify(after)) return 'none';
  if (before.connections.length !== after.connections.length || before.nodes.length !== after.nodes.length)
    return 'shader';

  const beforeConnections = JSON.stringify(before.connections);
  const afterConnections = JSON.stringify(after.connections);
  if (beforeConnections !== afterConnections) return 'shader';

  const beforeParameters = effectiveExposedParameterNodeIds(before);
  const afterParameters = effectiveExposedParameterNodeIds(after);
  const beforeById = new Map(before.nodes.map((node) => [node.id, node]));
  let changedParameterValue = false;
  for (const node of after.nodes) {
    const previous = beforeById.get(node.id);
    if (!previous || previous.type !== node.type) return 'shader';
    if (!sameParameterMetadata(previous, node)) return 'shader';
    if (previous.position[0] !== node.position[0] || previous.position[1] !== node.position[1]) return 'shader';

    if (JSON.stringify(previous.values) !== JSON.stringify(node.values)) {
      if (!beforeParameters.has(node.id) || !afterParameters.has(node.id)) return 'shader';
      changedParameterValue = true;
    }
  }

  return changedParameterValue ? 'parameter-values' : 'none';
};

export type MaterialEditorParameterKind = 'scalar' | 'vector' | 'color' | 'texture';

export type MaterialEditorParameter = {
  nodeId: string;
  name: string;
  type: MaterialGraphValueType;
  nodeType: MaterialGraphNodeType;
  editorKind: MaterialEditorParameterKind;
  range?: { min: number; max: number };
};

/**
 * Return authored exposed-parameter metadata for the inspector.
 *
 * Parameter type checking and reachability are owned exclusively by the native compiler; this
 * editor-side projection only filters obviously ineffective controls for presentation.
 * Only value-source nodes that contribute to Material Output are useful runtime parameters.
 * Disconnected nodes and operation/utility nodes are deliberately omitted so the editor never
 * presents controls that cannot affect the material result.
 */
export const materialEditorParameters = (graph: MaterialGraph): MaterialEditorParameter[] => {
  const effectiveParameters = effectiveExposedParameterNodeIds(graph);
  return graph.nodes.flatMap((node) => {
    if (!effectiveParameters.has(node.id) || !node.parameter) return [];
    const type: MaterialGraphValueType =
      node.type === 'textureSample' || node.type === 'textureSample2D'
        ? 'texture2d'
        : node.type === 'vector2'
          ? 'vec2'
          : node.type === 'vector3'
            ? 'vec3'
            : node.type === 'vector4' || node.type === 'colorRgba'
              ? 'vec4'
              : 'float';
    const editorKind: MaterialEditorParameterKind =
      node.type === 'textureSample' || node.type === 'textureSample2D'
        ? 'texture'
        : node.type === 'colorRgba'
          ? 'color'
          : type === 'float'
            ? 'scalar'
            : 'vector';
    return [
      {
        nodeId: node.id,
        name: node.parameter.name.trim() || 'Parameter',
        type,
        nodeType: node.type,
        editorKind,
        ...(node.type === 'constant' && materialScalarRange(node) ? { range: materialScalarRange(node)! } : {}),
      },
    ];
  });
};
