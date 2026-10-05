export type MaterialUpdateKind = 'parameters' | 'graph' | 'shader-source';

export type MaterialUpdateDecision = {
  recompileShader: boolean;
  refreshParameters: boolean;
  reason: string;
};

/**
 * Central policy for deciding whether an authored material update needs shader
 * compilation. Parameter values are runtime data and must not rebuild an
 * otherwise unchanged shader program.
 */
export function classifyMaterialUpdate(kind: MaterialUpdateKind): MaterialUpdateDecision {
  switch (kind) {
    case 'parameters':
      return {
        recompileShader: false,
        refreshParameters: true,
        reason: 'Parameter-only updates reuse the compiled material program.',
      };
    case 'graph':
      return {
        recompileShader: true,
        refreshParameters: true,
        reason: 'Graph topology or authored shader semantics changed.',
      };
    case 'shader-source':
      return {
        recompileShader: true,
        refreshParameters: true,
        reason: 'Custom shader source changed.',
      };
  }
}
