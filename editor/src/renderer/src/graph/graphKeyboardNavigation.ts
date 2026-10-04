export type GraphNavigationDirection = 'left' | 'right' | 'up' | 'down';

export interface GraphNavigationNode {
  id: string;
  x: number;
  y: number;
}

function isFiniteNode(node: GraphNavigationNode): boolean {
  return Number.isFinite(node.x) && Number.isFinite(node.y);
}

function isInDirection(
  dx: number,
  dy: number,
  direction: GraphNavigationDirection,
): boolean {
  switch (direction) {
    case 'left':
      return dx < 0;
    case 'right':
      return dx > 0;
    case 'up':
      return dy < 0;
    case 'down':
      return dy > 0;
  }
}

function directionalScore(
  dx: number,
  dy: number,
  direction: GraphNavigationDirection,
): number {
  const primary = direction === 'left' || direction === 'right' ? Math.abs(dx) : Math.abs(dy);
  const secondary = direction === 'left' || direction === 'right' ? Math.abs(dy) : Math.abs(dx);

  // Prefer candidates in the requested direction, then penalize cross-axis travel so
  // arrow-key navigation follows the visual row/column before jumping diagonally.
  return primary + secondary * 2;
}

/**
 * Returns the visually nearest node in an arrow-key direction.
 *
 * The helper is deliberately domain-neutral: Material, Flow, and future graph editors
 * only need to provide stable node ids and graph-space positions. Invalid/stale
 * positions are ignored and ties are resolved by stable id for deterministic behavior.
 */
export function findGraphNavigationTarget(
  nodes: readonly GraphNavigationNode[],
  currentId: string,
  direction: GraphNavigationDirection,
): string | null {
  const current = nodes.find((node) => node.id === currentId && isFiniteNode(node));
  if (!current) {
    return null;
  }

  let best: GraphNavigationNode | null = null;
  let bestScore = Number.POSITIVE_INFINITY;

  for (const candidate of nodes) {
    if (candidate.id === current.id || !isFiniteNode(candidate)) {
      continue;
    }

    const dx = candidate.x - current.x;
    const dy = candidate.y - current.y;
    if (!isInDirection(dx, dy, direction)) {
      continue;
    }

    const score = directionalScore(dx, dy, direction);
    if (
      score < bestScore ||
      (score === bestScore && best !== null && candidate.id.localeCompare(best.id) < 0)
    ) {
      best = candidate;
      bestScore = score;
    }
  }

  return best?.id ?? null;
}
