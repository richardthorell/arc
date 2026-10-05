import { describe, expect, it } from 'vitest';

import { classifyMaterialUpdate } from './materialUpdatePolicy';

describe('material update policy', () => {
  it('does not rebuild shaders for parameter-only edits', () => {
    expect(classifyMaterialUpdate('parameters')).toEqual({
      recompileShader: false,
      refreshParameters: true,
      reason: 'Parameter-only updates reuse the compiled material program.',
    });
  });

  it.each(['graph', 'shader-source'] as const)('recompiles for %s changes', (kind) => {
    const decision = classifyMaterialUpdate(kind);
    expect(decision.recompileShader).toBe(true);
    expect(decision.refreshParameters).toBe(true);
  });
});
