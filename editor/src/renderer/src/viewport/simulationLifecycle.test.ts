import { describe, expect, it } from 'vitest';
import { transitionRuntimeSession } from './simulationLifecycle';

describe('simulation lifecycle', () => {
  it('starts Play and Simulate as explicit isolated runtime modes', () => {
    expect(transitionRuntimeSession({ phase: 'authoring' }, { type: 'start', mode: 'play' })).toEqual({
      phase: 'running',
      mode: 'play',
    });
    expect(transitionRuntimeSession({ phase: 'authoring' }, { type: 'start', mode: 'simulate' })).toEqual({
      phase: 'running',
      mode: 'simulate',
    });
  });

  it('preserves mode across pause and resume', () => {
    const paused = transitionRuntimeSession({ phase: 'running', mode: 'simulate' }, { type: 'pause' });
    expect(paused).toEqual({ phase: 'paused', mode: 'simulate' });
    expect(transitionRuntimeSession(paused, { type: 'resume' })).toEqual({ phase: 'running', mode: 'simulate' });
  });

  it('switches between Play and Simulate without returning to authoring', () => {
    expect(transitionRuntimeSession({ phase: 'running', mode: 'simulate' }, { type: 'switch-mode', mode: 'play' })).toEqual({
      phase: 'running',
      mode: 'play',
    });
    expect(transitionRuntimeSession({ phase: 'paused', mode: 'play' }, { type: 'switch-mode', mode: 'simulate' })).toEqual({
      phase: 'paused',
      mode: 'simulate',
    });
  });

  it('returns to authoring only when the runtime session stops', () => {
    expect(transitionRuntimeSession({ phase: 'running', mode: 'simulate' }, { type: 'stop' })).toEqual({ phase: 'authoring' });
  });

  it('rejects invalid lifecycle transitions', () => {
    expect(() => transitionRuntimeSession({ phase: 'authoring' }, { type: 'pause' })).toThrow();
    expect(() => transitionRuntimeSession({ phase: 'authoring' }, { type: 'switch-mode', mode: 'simulate' })).toThrow();
    expect(() => transitionRuntimeSession({ phase: 'running', mode: 'play' }, { type: 'start', mode: 'simulate' })).toThrow();
  });
});
