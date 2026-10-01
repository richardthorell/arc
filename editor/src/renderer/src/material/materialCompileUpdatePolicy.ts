import { materialGraphEditImpact, type MaterialGraphEditImpact } from './materialCompiler';
import type { MaterialGraph } from './materialGraphTypes';

export type MaterialCompileUpdateAction = 'none' | 'parameter-update' | 'shader-compile';

/**
 * Decide how an authored graph edit should update a live material.
 *
 * Exposed parameter value edits are runtime data updates and must not invalidate the
 * last successful shader compilation. Shader-affecting edits continue through ARC's
 * authoritative native Material IR/compiler path.
 */
export const materialCompileUpdateAction = (before: MaterialGraph, after: MaterialGraph): MaterialCompileUpdateAction =>
  materialCompileUpdateActionForImpact(materialGraphEditImpact(before, after));

export const materialCompileUpdateActionForImpact = (impact: MaterialGraphEditImpact): MaterialCompileUpdateAction => {
  switch (impact) {
    case 'none':
      return 'none';
    case 'parameter-values':
      return 'parameter-update';
    case 'shader':
      return 'shader-compile';
  }
};

/** Only shader-affecting edits invalidate compiled shader state or schedule compilation. */
export const materialEditRequiresShaderCompile = (before: MaterialGraph, after: MaterialGraph): boolean =>
  materialCompileUpdateAction(before, after) === 'shader-compile';
