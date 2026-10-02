import { describe, expect, it } from 'vitest';

import {
  attachFlowDebugger,
  createFlowDebuggerState,
  detachFlowDebugger,
  isFlowBreakpointEnabled,
  recordFlowDebugPause,
  recordFlowDebugWatches,
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

  it('accepts watch snapshots only from the attached runtime instance', () => {
    const attached = instance('entity-a', 'runtime-1');
    const stale = instance('entity-a', 'runtime-old');
    let state = attachFlowDebugger(createFlowDebuggerState(), attached);

    state = recordFlowDebugWatches(state, {
      instance: stale,
      watches: [{ slotId: 'health', label: 'Health', value: 1 }],
    });
    expect(state.watches).toEqual([]);

    state = recordFlowDebugWatches(state, {
      instance: attached,
      watches: [
        { slotId: ' health ', label: 'Health', value: 100 },
        { slotId: 'health', label: 'Duplicate', value: 50 },
        { slotId: '', label: 'Invalid', value: true },
        { slotId: 'isGrounded', label: 'Is Grounded', value: false },
      ],
    });

    expect(state.watches).toEqual([
      { slotId: 'health', label: 'Health', value: 100 },
      { slotId: 'isGrounded', label: 'Is Grounded', value: false },
    ]);
  });

  it('clears runtime watch values when attachment identity changes or detaches', () => {
    const first = instance('entity-a', 'runtime-1');
    const restarted = instance('entity-a', 'runtime-2');
    let state = attachFlowDebugger(createFlowDebuggerState(), first);
    state = recordFlowDebugWatches(state, {
      instance: first,
      watches: [{ slotId: 'speed', label: 'Speed', value: 3.5 }],
    });

    state = attachFlowDebugger(state, restarted);
    expect(state.watches).toEqual([]);

    state = recordFlowDebugWatches(state, {
      instance: restarted,
      watches: [{ slotId: 'speed', label: 'Speed', value: 4 }],
    });
    state = detachFlowDebugger(state);
    expect(state.watches).toEqual([]);
  });
});
