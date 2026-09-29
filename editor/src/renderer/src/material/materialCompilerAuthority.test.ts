import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const materialCompilerSource = readFileSync(fileURLToPath(new URL('./materialCompiler.ts', import.meta.url)), 'utf8');

/**
 * The renderer may adapt native diagnostics and derive authoring-only presentation metadata,
 * but material type checking, reachability, IR generation, and shader compilation belong to
 * the native material compiler. Keep this guard close to the editor adapter so a future
 * compatibility evaluator cannot quietly become a second compiler path.
 */
describe('material compiler authority', () => {
  it('keeps the renderer adapter free of graph compile/evaluate entry points', () => {
    const forbiddenEntryPoints = [
      /export\s+(?:const|function)\s+compileMaterialGraph\b/,
      /export\s+(?:const|function)\s+evaluateMaterialGraph\b/,
      /export\s+(?:const|function)\s+compileMaterialNode\b/,
      /export\s+(?:const|function)\s+evaluateMaterialNode\b/,
    ];

    for (const pattern of forbiddenEntryPoints) expect(materialCompilerSource).not.toMatch(pattern);
  });

  it('documents native ownership beside editor-only parameter projection', () => {
    expect(materialCompilerSource).toContain("Diagnostic returned by ARC's native Material IR/compiler pipeline.");
    expect(materialCompilerSource).toContain('owned exclusively by the native compiler');
    expect(materialCompilerSource).toContain('materialEditorParameters');
  });
});
