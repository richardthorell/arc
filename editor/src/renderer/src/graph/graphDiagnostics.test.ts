import { describe, expect, it } from 'vitest';

import {
  graphDiagnosticNodeId,
  graphDiagnosticsForNode,
  graphDiagnosticTargetKey,
  sortGraphDiagnostics,
  summarizeGraphDiagnostics,
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

  it('builds stable target keys for domain-neutral overlay selection', () => {
    expect(graphDiagnosticTargetKey(diagnostics[0])).toBe('pin:node-b:input');
    expect(graphDiagnosticTargetKey(diagnostics[1])).toBe('node:node-a');
    expect(
      graphDiagnosticTargetKey({
        id: 'connection',
        severity: 'error',
        message: 'Invalid connection',
        target: { kind: 'connection', connectionId: 'wire-1' },
      }),
    ).toBe('connection:wire-1');
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

  it('summarizes diagnostics per node for shared badges and detail surfaces', () => {
    const summaries = summarizeGraphDiagnostics([
      ...diagnostics,
      {
        id: 'connection-b',
        severity: 'error',
        message: 'Connection is invalid',
        target: { kind: 'connection', connectionId: 'wire-b', sourceNodeId: 'node-a', targetNodeId: 'node-b' },
      },
      {
        id: 'orphan-wire',
        severity: 'warning',
        message: 'Wire has no resolvable node',
        target: { kind: 'connection', connectionId: 'orphan' },
      },
    ]);

    expect(summaries.map((summary) => summary.nodeId)).toEqual(['node-a', 'node-b']);
    expect(summaries[0]).toMatchObject({
      nodeId: 'node-a',
      highestSeverity: 'error',
      count: 2,
      errorCount: 1,
      warningCount: 0,
      infoCount: 1,
    });
    expect(summaries[1]).toMatchObject({
      nodeId: 'node-b',
      highestSeverity: 'error',
      count: 2,
      errorCount: 1,
      warningCount: 1,
      infoCount: 0,
    });
    expect(summaries[1].diagnostics.map((diagnostic) => diagnostic.id)).toEqual(['connection-b', 'warning-b']);
  });
});
