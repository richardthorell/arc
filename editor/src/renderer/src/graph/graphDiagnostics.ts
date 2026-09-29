export type GraphDiagnosticSeverity = 'error' | 'warning' | 'info';

export type GraphDiagnosticTarget =
  | { kind: 'node'; nodeId: string }
  | { kind: 'pin'; nodeId: string; pinId: string }
  | { kind: 'connection'; connectionId: string; sourceNodeId?: string; targetNodeId?: string };

export type GraphDiagnostic = {
  id: string;
  severity: GraphDiagnosticSeverity;
  message: string;
  details?: string;
  target: GraphDiagnosticTarget;
};

export type GraphDiagnosticSummary = {
  nodeId: string;
  highestSeverity: GraphDiagnosticSeverity;
  count: number;
  errorCount: number;
  warningCount: number;
  infoCount: number;
  diagnostics: GraphDiagnostic[];
};

const severityRank: Record<GraphDiagnosticSeverity, number> = {
  error: 0,
  warning: 1,
  info: 2,
};

export const graphDiagnosticNodeId = (diagnostic: GraphDiagnostic): string | undefined => {
  switch (diagnostic.target.kind) {
    case 'node':
    case 'pin':
      return diagnostic.target.nodeId;
    case 'connection':
      return diagnostic.target.targetNodeId ?? diagnostic.target.sourceNodeId;
  }
};

export const graphDiagnosticTargetKey = (diagnostic: GraphDiagnostic): string => {
  switch (diagnostic.target.kind) {
    case 'node':
      return `node:${diagnostic.target.nodeId}`;
    case 'pin':
      return `pin:${diagnostic.target.nodeId}:${diagnostic.target.pinId}`;
    case 'connection':
      return `connection:${diagnostic.target.connectionId}`;
  }
};

export const sortGraphDiagnostics = (diagnostics: readonly GraphDiagnostic[]): GraphDiagnostic[] =>
  [...diagnostics].sort((left, right) => {
    const severity = severityRank[left.severity] - severityRank[right.severity];
    if (severity !== 0) return severity;

    const leftNode = graphDiagnosticNodeId(left) ?? '';
    const rightNode = graphDiagnosticNodeId(right) ?? '';
    const node = leftNode.localeCompare(rightNode);
    if (node !== 0) return node;

    return left.id.localeCompare(right.id);
  });

export const graphDiagnosticsForNode = (diagnostics: readonly GraphDiagnostic[], nodeId: string): GraphDiagnostic[] =>
  sortGraphDiagnostics(diagnostics.filter((diagnostic) => graphDiagnosticNodeId(diagnostic) === nodeId));

export const summarizeGraphDiagnostics = (diagnostics: readonly GraphDiagnostic[]): GraphDiagnosticSummary[] => {
  const byNode = new Map<string, GraphDiagnostic[]>();

  for (const diagnostic of diagnostics) {
    const nodeId = graphDiagnosticNodeId(diagnostic);
    if (!nodeId) continue;
    const existing = byNode.get(nodeId);
    if (existing) existing.push(diagnostic);
    else byNode.set(nodeId, [diagnostic]);
  }

  return [...byNode.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([nodeId, nodeDiagnostics]) => {
      const sorted = sortGraphDiagnostics(nodeDiagnostics);
      return {
        nodeId,
        highestSeverity: sorted[0].severity,
        count: sorted.length,
        errorCount: sorted.filter((diagnostic) => diagnostic.severity === 'error').length,
        warningCount: sorted.filter((diagnostic) => diagnostic.severity === 'warning').length,
        infoCount: sorted.filter((diagnostic) => diagnostic.severity === 'info').length,
        diagnostics: sorted,
      };
    });
};
