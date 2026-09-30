export type MaterialPermutationCompileState = 'cached' | 'compiling' | 'failed' | 'missing';

export type MaterialPermutationVariantStatus = {
  key: string;
  state: MaterialPermutationCompileState;
};

export type MaterialPermutationCompileSummary = {
  total: number;
  cached: number;
  compiling: number;
  failed: number;
  missing: number;
  complete: boolean;
};

/**
 * Summarize compiler/cache state for a material's deterministic permutation keys.
 * Duplicate keys are ignored so editor presentation cannot inflate cooker-owned variant counts.
 */
export const summarizeMaterialPermutationCompileStatus = (
  variants: readonly MaterialPermutationVariantStatus[],
): MaterialPermutationCompileSummary => {
  const byKey = new Map<string, MaterialPermutationCompileState>();
  for (const variant of variants) {
    if (!variant.key || byKey.has(variant.key)) continue;
    byKey.set(variant.key, variant.state);
  }

  const summary: MaterialPermutationCompileSummary = {
    total: byKey.size,
    cached: 0,
    compiling: 0,
    failed: 0,
    missing: 0,
    complete: false,
  };

  for (const state of byKey.values()) summary[state] += 1;
  summary.complete = summary.total > 0 && summary.cached === summary.total;
  return summary;
};
