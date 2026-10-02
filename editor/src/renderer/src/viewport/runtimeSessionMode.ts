export type RuntimeSessionMode = 'edit' | 'simulate' | 'play';

export type RuntimeSessionState = {
  mode: RuntimeSessionMode;
  runtimeWorldId: string | null;
  paused: boolean;
};

export type RuntimeSessionEffect =
  | 'begin-runtime-world'
  | 'end-runtime-world'
  | 'acquire-player-control'
  | 'release-player-control';

export type RuntimeSessionTransition = {
  state: RuntimeSessionState;
  effects: readonly RuntimeSessionEffect[];
};

export const createEditSessionState = (): RuntimeSessionState => ({
  mode: 'edit',
  runtimeWorldId: null,
  paused: false,
});

const requireRuntimeWorldId = (runtimeWorldId: string): string => {
  const normalized = runtimeWorldId.trim();
  if (!normalized) throw new Error('Runtime session requires a non-empty isolated runtime world ID');
  return normalized;
};

/**
 * Starts or changes the editor runtime mode without changing the authoring world.
 *
 * Entering Simulate/Play from Edit creates one isolated runtime world. Switching
 * between Simulate and Play keeps that world alive and only changes possession,
 * so Begin/End Play-style lifecycle callbacks remain paired at the session edge.
 */
export const transitionRuntimeSession = (
  current: RuntimeSessionState,
  target: RuntimeSessionMode,
  runtimeWorldId?: string,
): RuntimeSessionTransition => {
  if (current.mode === target) return { state: current, effects: [] };

  if (current.mode === 'edit') {
    if (target === 'edit') return { state: current, effects: [] };
    const worldId = requireRuntimeWorldId(runtimeWorldId ?? '');
    return {
      state: { mode: target, runtimeWorldId: worldId, paused: false },
      effects: target === 'play' ? ['begin-runtime-world', 'acquire-player-control'] : ['begin-runtime-world'],
    };
  }

  if (target === 'edit') {
    return {
      state: createEditSessionState(),
      effects:
        current.mode === 'play'
          ? ['release-player-control', 'end-runtime-world']
          : ['end-runtime-world'],
    };
  }

  if (!current.runtimeWorldId) throw new Error('Active runtime session is missing its isolated runtime world ID');

  return {
    state: { ...current, mode: target },
    effects: target === 'play' ? ['acquire-player-control'] : ['release-player-control'],
  };
};

export const setRuntimeSessionPaused = (current: RuntimeSessionState, paused: boolean): RuntimeSessionState => {
  if (current.mode === 'edit') return current;
  return current.paused === paused ? current : { ...current, paused };
};

export const usesEditorNavigation = (state: RuntimeSessionState): boolean => state.mode !== 'play';
