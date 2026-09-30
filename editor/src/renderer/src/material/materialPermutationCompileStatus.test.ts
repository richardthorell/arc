import { describe, expect, it } from 'vitest';

import { summarizeMaterialPermutationCompileStatus } from './materialPermutationCompileStatus';

describe('summarizeMaterialPermutationCompileStatus', () => {
  it('summarizes deterministic compiler/cache state', () => {
    expect(
      summarizeMaterialPermutationCompileStatus([
        { key: 'base', state: 'cached' },
        { key: 'normal-map', state: 'compiling' },
        { key: 'clear-coat', state: 'failed' },
        { key: 'quality-high', state: 'missing' },
      ]),
    ).toEqual({ total: 4, cached: 1, compiling: 1, failed: 1, missing: 1, complete: false });
  });

  it('deduplicates stable permutation keys', () => {
    expect(
      summarizeMaterialPermutationCompileStatus([
        { key: 'base', state: 'cached' },
        { key: 'base', state: 'failed' },
      ]),
    ).toEqual({ total: 1, cached: 1, compiling: 0, failed: 0, missing: 0, complete: true });
  });

  it('does not report an empty material as completely cached', () => {
    expect(summarizeMaterialPermutationCompileStatus([])).toEqual({
      total: 0,
      cached: 0,
      compiling: 0,
      failed: 0,
      missing: 0,
      complete: false,
    });
  });
});
