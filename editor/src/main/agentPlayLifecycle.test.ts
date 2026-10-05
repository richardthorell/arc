import { describe, expect, it } from 'vitest';

import { planPlayLifecycleCommand } from './agentPlayLifecycle';

describe('agent Play lifecycle contract', () => {
  it('uses the Play World for play and resume transitions', () => {
    expect(planPlayLifecycleCommand('stopped', { kind: 'play' })).toEqual({
      command: 'play',
      from: 'stopped',
      expectedState: 'playing',
      scope: 'play-world',
    });
    expect(planPlayLifecycleCommand('paused', { kind: 'play' })).toEqual({
      command: 'play',
      from: 'paused',
      expectedState: 'playing',
      scope: 'play-world',
    });
  });

  it('only pauses a running Play World', () => {
    expect(planPlayLifecycleCommand('playing', { kind: 'pause' })).toMatchObject({
      command: 'pause',
      expectedState: 'paused',
      scope: 'play-world',
    });
    expect(() => planPlayLifecycleCommand('stopped', { kind: 'pause' })).toThrow(
      'Pause requires a running Play World',
    );
  });

  it('makes stop explicitly terminate an active Play World', () => {
    expect(planPlayLifecycleCommand('paused', { kind: 'stop' })).toMatchObject({
      command: 'stop',
      expectedState: 'stopped',
      scope: 'play-world',
    });
    expect(() => planPlayLifecycleCommand('stopped', { kind: 'stop' })).toThrow('Play is already stopped');
  });

  it('only steps fixed positive tick counts while paused', () => {
    expect(planPlayLifecycleCommand('paused', { kind: 'step', ticks: 3 })).toEqual({
      command: 'step',
      from: 'paused',
      expectedState: 'paused',
      ticks: 3,
      scope: 'play-world',
    });
    expect(() => planPlayLifecycleCommand('playing', { kind: 'step', ticks: 1 })).toThrow(
      'Step requires a paused Play World',
    );
    expect(() => planPlayLifecycleCommand('paused', { kind: 'step', ticks: 0 })).toThrow(
      'Step ticks must be a positive safe integer',
    );
    expect(() => planPlayLifecycleCommand('paused', { kind: 'step', ticks: 1.5 })).toThrow(
      'Step ticks must be a positive safe integer',
    );
  });

  it('rejects duplicate play while already running', () => {
    expect(() => planPlayLifecycleCommand('playing', { kind: 'play' })).toThrow('Play is already running');
  });
});
