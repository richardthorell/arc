export type FlowDebugInstanceId = {
  entityId: string;
  graphId: string;
  runtimeInstanceId: string;
};

export type FlowDebugLocation = {
  nodeId: string;
  pinId?: string;
};

export type FlowBreakpoint = {
  graphId: string;
  nodeId: string;
  enabled: boolean;
};

export type FlowDebugPause = {
  instance: FlowDebugInstanceId;
  location: FlowDebugLocation;
  reason: 'breakpoint' | 'step' | 'error' | 'instruction_budget';
};

export type FlowDebugValue = string | number | boolean | null;

export type FlowDebugWatch = {
  slotId: string;
  label: string;
  value: FlowDebugValue;
};

export type FlowDebugWatchSnapshot = {
  instance: FlowDebugInstanceId;
  watches: readonly FlowDebugWatch[];
};

export type FlowDebuggerState = {
  attachedInstance: FlowDebugInstanceId | null;
  breakpoints: readonly FlowBreakpoint[];
  pause: FlowDebugPause | null;
  watches: readonly FlowDebugWatch[];
};

export const createFlowDebuggerState = (): FlowDebuggerState => ({
  attachedInstance: null,
  breakpoints: [],
  pause: null,
  watches: [],
});

const sameInstance = (left: FlowDebugInstanceId, right: FlowDebugInstanceId) =>
  left.entityId === right.entityId &&
  left.graphId === right.graphId &&
  left.runtimeInstanceId === right.runtimeInstanceId;

export function attachFlowDebugger(state: FlowDebuggerState, instance: FlowDebugInstanceId): FlowDebuggerState {
  if (state.attachedInstance && sameInstance(state.attachedInstance, instance)) return state;
  return { ...state, attachedInstance: instance, pause: null, watches: [] };
}

export function detachFlowDebugger(state: FlowDebuggerState): FlowDebuggerState {
  if (!state.attachedInstance && !state.pause && state.watches.length === 0) return state;
  return { ...state, attachedInstance: null, pause: null, watches: [] };
}

export function setFlowBreakpoint(
  state: FlowDebuggerState,
  graphId: string,
  nodeId: string,
  enabled: boolean,
): FlowDebuggerState {
  const index = state.breakpoints.findIndex(
    (breakpoint) => breakpoint.graphId === graphId && breakpoint.nodeId === nodeId,
  );

  if (index < 0) {
    return { ...state, breakpoints: [...state.breakpoints, { graphId, nodeId, enabled }] };
  }

  if (state.breakpoints[index]?.enabled === enabled) return state;
  const breakpoints = [...state.breakpoints];
  breakpoints[index] = { graphId, nodeId, enabled };
  return { ...state, breakpoints };
}

export function removeFlowBreakpoint(state: FlowDebuggerState, graphId: string, nodeId: string): FlowDebuggerState {
  const breakpoints = state.breakpoints.filter(
    (breakpoint) => breakpoint.graphId !== graphId || breakpoint.nodeId !== nodeId,
  );
  return breakpoints.length === state.breakpoints.length ? state : { ...state, breakpoints };
}

export function recordFlowDebugPause(state: FlowDebuggerState, pause: FlowDebugPause): FlowDebuggerState {
  if (!state.attachedInstance || !sameInstance(state.attachedInstance, pause.instance)) return state;
  return { ...state, pause };
}

export function recordFlowDebugWatches(
  state: FlowDebuggerState,
  snapshot: FlowDebugWatchSnapshot,
): FlowDebuggerState {
  if (!state.attachedInstance || !sameInstance(state.attachedInstance, snapshot.instance)) return state;

  const seen = new Set<string>();
  const watches: FlowDebugWatch[] = [];
  for (const watch of snapshot.watches) {
    const slotId = watch.slotId.trim();
    if (!slotId || seen.has(slotId)) continue;
    seen.add(slotId);
    watches.push({ ...watch, slotId });
  }

  return { ...state, watches };
}

export function resumeFlowDebugger(state: FlowDebuggerState): FlowDebuggerState {
  return state.pause ? { ...state, pause: null } : state;
}

export function isFlowBreakpointEnabled(state: FlowDebuggerState, graphId: string, nodeId: string): boolean {
  return state.breakpoints.some(
    (breakpoint) => breakpoint.graphId === graphId && breakpoint.nodeId === nodeId && breakpoint.enabled,
  );
}
