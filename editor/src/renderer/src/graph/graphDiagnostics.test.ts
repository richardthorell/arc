import { describe, expect, it } from 'vitest';

import {
  graphDiagnosticNodeId,
  graphDiagnosticsForNode,
  sortGraphDiagnostics,
  type GraphDiagnostic,
} from './graphDiagnostics';

const diagnostics: GraphDiagnostic[] = [
  {
    id: 'warning-b',
    severity: 'warning',
    message: 'Input is unused',
    target: { kind: 'pin', nodeId: 'node-b', pinId: 'input' },
  },
  {
    id: 'error-a',
    severity: 'error',
    message: 'Type mismatch',
    target: { kind: 'node', nodeId: 'node-a' },
  },
  {
    id: 'info-a',
    severity: 'info',
    message: 'Constant folded',
    target: { kind: 'node', nodeId: 'node-a' },
  },
];

describe('graphDiagnostics', () => {
  it('resolves node targets for node, pin, and connection diagnostics', () => {
    expect(graphDiagnosticNodeId(diagnostics[0])).toBe('node-b');
    expect(
      graphDiagnosticNodeId({
        id: 'connection',
        severity: 'error',
        message: 'Invalid connection',
        target: { kind: 'connection', connectionId: 'wire-1', sourceNodeId: 'source', targetNodeId: 'target' },
      }),
    ).toBe('target');
  });

  it('sorts deterministically by severity, node, and stable diagnostic id without mutating input', () => {
    const original = [...diagnostics];
    expect(sortGraphDiagnostics(diagnostics).map((diagnostic) => diagnostic.id)).toEqual([
      'error-a',
      'warning-b',
      'info-a',
    ]);
    expect(diagnostics).toEqual(original);
  });

  it('returns only diagnostics that can jump to the requested node', () => {
    expect(graphDiagnosticsForNode(diagnostics, 'node-a').map((diagnostic) => diagnostic.id)).toEqual([
      'error-a',
      'info-a',
    ]);
  });
});
