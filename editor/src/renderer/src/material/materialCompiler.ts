import type { MaterialGraph, MaterialGraphValueType } from './materialGraphTypes';

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
      node.type === 'vector2'
        ? 'vec2'
        : node.type === 'vector3' || node.type === 'colorRgb'
          ? 'vec3'
          : node.type === 'vector4' || node.type === 'colorRgba'
            ? 'vec4'
            : 'float';
    return [{ nodeId: node.id, name: node.parameter.name.trim() || 'Parameter', type }];
  });
