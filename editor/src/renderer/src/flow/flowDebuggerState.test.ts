import { describe, expect, it } from 'vitest';

import {
  attachFlowDebugger,
  createFlowDebuggerState,
  isFlowBreakpointEnabled,
  recordFlowDebugPause,
  removeFlowBreakpoint,
  resumeFlowDebugger,
  setFlowBreakpoint,
} from './flowDebuggerState';

const instance = (entityId: string, runtimeInstanceId = `runtime-${entityId}`) => ({
  entityId,
  graphId: 'graph-player',
  runtimeInstanceId,
});

describe('flowDebuggerState', () => {
  it('keeps pauses scoped to the attached entity runtime instance', () => {
    const attached = instance('entity-a');
    const other = instance('entity-b');
    let state = attachFlowDebugger(createFlowDebuggerState(), attached);

    state = recordFlowDebugPause(state, {
      instance: other,
      location: { nodeId: 'node-other' },
      reason: 'breakpoint',
    });
    expect(state.pause).toBeNull();

    state = recordFlowDebugPause(state, {
      instance: attached,
      location: { nodeId: 'node-a', pinId: 'out' },
      reason: 'step',
    });
    expect(state.pause?.location).toEqual({ nodeId: 'node-a', pinId: 'out' });

    state = resumeFlowDebugger(state);
    expect(state.pause).toBeNull();
  });

  it('does not conflate two runtime instances for the same entity and graph', () => {
    const first = instance('entity-a', 'runtime-1');
    const restarted = instance('entity-a', 'runtime-2');
    let state = attachFlowDebugger(createFlowDebuggerState(), restarted);

    state = recordFlowDebugPause(state, {
      instance: first,
      location: { nodeId: 'stale-node' },
      reason: 'error',
    });

    expect(state.pause).toBeNull();
  });

  it('stores breakpoints by graph and node independently of the attached instance', () => {
    let state = createFlowDebuggerState();
    state = setFlowBreakpoint(state, 'graph-player', 'node-start', true);
    state = setFlowBreakpoint(state, 'graph-enemy', 'node-start', false);

    expect(isFlowBreakpointEnabled(state, 'graph-player', 'node-start')).toBe(true);
    expect(isFlowBreakpointEnabled(state, 'graph-enemy', 'node-start')).toBe(false);

    state = setFlowBreakpoint(state, 'graph-enemy', 'node-start', true);
    expect(isFlowBreakpointEnabled(state, 'graph-enemy', 'node-start')).toBe(true);

    state = removeFlowBreakpoint(state, 'graph-player', 'node-start');
    expect(isFlowBreakpointEnabled(state, 'graph-player', 'node-start')).toBe(false);
    expect(isFlowBreakpointEnabled(state, 'graph-enemy', 'node-start')).toBe(true);
  });
});
