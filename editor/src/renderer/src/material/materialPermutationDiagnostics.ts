export interface MaterialStaticSwitch {
  id: string;
  name: string;
  /** Number of compile-time values this switch can produce. Boolean switches use 2. */
  variantCount?: number;
  /** A fixed switch does not create another compiled permutation. */
  fixed?: boolean;
}

export type MaterialPermutationSeverity = 'normal' | 'warning' | 'critical';

export interface MaterialPermutationCause {
  id: string;
  name: string;
  variantCount: number;
}

export interface MaterialPermutationDiagnostics {
  permutationCount: number;
  severity: MaterialPermutationSeverity;
  causes: MaterialPermutationCause[];
  message: string;
}

export interface MaterialPermutationLimits {
  warning: number;
  critical: number;
}

export const DEFAULT_MATERIAL_PERMUTATION_LIMITS: MaterialPermutationLimits = {
  warning: 32,
  critical: 128,
};

function normalizedVariantCount(value: number | undefined): number {
  if (value === undefined) return 2;
  if (!Number.isSafeInteger(value) || value < 1) return 1;
  return value;
}

/**
 * Produces a deterministic diagnostic for authored compile-time material switches.
 * Duplicate stable IDs are counted once so editor/cooker callers cannot accidentally
 * inflate the reported permutation space while reconciling authored metadata.
 */
export function diagnoseMaterialPermutations(
  switches: readonly MaterialStaticSwitch[],
  limits: MaterialPermutationLimits = DEFAULT_MATERIAL_PERMUTATION_LIMITS,
): MaterialPermutationDiagnostics {
  const unique = new Map<string, MaterialPermutationCause>();

  for (const entry of switches) {
    if (entry.fixed || unique.has(entry.id)) continue;
    const variantCount = normalizedVariantCount(entry.variantCount);
    if (variantCount <= 1) continue;
    unique.set(entry.id, { id: entry.id, name: entry.name, variantCount });
  }

  const causes = [...unique.values()].sort((left, right) => left.id.localeCompare(right.id, 'en'));

  let permutationCount = 1;
  for (const cause of causes) {
    if (permutationCount > Number.MAX_SAFE_INTEGER / cause.variantCount) {
      permutationCount = Number.MAX_SAFE_INTEGER;
      break;
    }
    permutationCount *= cause.variantCount;
  }

  const severity: MaterialPermutationSeverity =
    permutationCount >= limits.critical ? 'critical' : permutationCount >= limits.warning ? 'warning' : 'normal';

  const causeSummary = causes.length
    ? causes.map((cause) => `${cause.name} (${cause.variantCount}x)`).join(', ')
    : 'No compile-time switches';

  return {
    permutationCount,
    severity,
    causes,
    message: `${permutationCount} shader permutation${permutationCount === 1 ? '' : 's'} · ${causeSummary}`,
  };
}
