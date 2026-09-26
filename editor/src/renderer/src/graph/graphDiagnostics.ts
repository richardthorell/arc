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

export const graphDiagnosticsForNode = (
  diagnostics: readonly GraphDiagnostic[],
  nodeId: string,
): GraphDiagnostic[] => sortGraphDiagnostics(diagnostics.filter((diagnostic) => graphDiagnosticNodeId(diagnostic) === nodeId));
