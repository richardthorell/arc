import { describe, expect, it } from 'vitest';

import {
  createEditSessionState,
  setRuntimeSessionPaused,
  transitionRuntimeSession,
  usesEditorNavigation,
} from './runtimeSessionMode';

describe('runtime session mode', () => {
  it('starts Simulate in an isolated runtime world while retaining editor navigation', () => {
    const transition = transitionRuntimeSession(createEditSessionState(), 'simulate', 'runtime-42');

    expect(transition).toEqual({
      state: { mode: 'simulate', runtimeWorldId: 'runtime-42', paused: false },
      effects: ['begin-runtime-world'],
    });
    expect(usesEditorNavigation(transition.state)).toBe(true);
  });

  it('switches between Simulate and Play without restarting the runtime world', () => {
    const simulated = transitionRuntimeSession(createEditSessionState(), 'simulate', 'runtime-42');
    const played = transitionRuntimeSession(simulated.state, 'play');
    const simulatedAgain = transitionRuntimeSession(played.state, 'simulate');

    expect(played.state.runtimeWorldId).toBe('runtime-42');
    expect(played.effects).toEqual(['acquire-player-control']);
    expect(usesEditorNavigation(played.state)).toBe(false);
    expect(simulatedAgain.state.runtimeWorldId).toBe('runtime-42');
    expect(simulatedAgain.effects).toEqual(['release-player-control']);
    expect(usesEditorNavigation(simulatedAgain.state)).toBe(true);
  });

  it('pairs runtime lifecycle at the session edge when stopping', () => {
    const played = transitionRuntimeSession(createEditSessionState(), 'play', 'runtime-7');
    const stopped = transitionRuntimeSession(played.state, 'edit');

    expect(played.effects).toEqual(['begin-runtime-world', 'acquire-player-control']);
    expect(stopped).toEqual({
      state: createEditSessionState(),
      effects: ['release-player-control', 'end-runtime-world'],
    });
  });

  it('supports pause state in both runtime modes without changing Edit state', () => {
    const simulated = transitionRuntimeSession(createEditSessionState(), 'simulate', 'runtime-1').state;
    expect(setRuntimeSessionPaused(simulated, true)).toMatchObject({ mode: 'simulate', paused: true });
    expect(setRuntimeSessionPaused(createEditSessionState(), true)).toEqual(createEditSessionState());
  });

  it('rejects runtime starts without an isolated world identity', () => {
    expect(() => transitionRuntimeSession(createEditSessionState(), 'simulate', '   ')).toThrow(/runtime world ID/);
  });
});
