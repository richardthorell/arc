import { describe, expect, it } from 'vitest';

import { validateAgentPlayControlRequest } from './agentPlayControlContract';

describe('agent Play control contract', () => {
  it('accepts lifecycle transitions from their valid states', () => {
    expect(validateAgentPlayControlRequest({ action: 'play' }, 'stopped')).toEqual({ action: 'play' });
    expect(validateAgentPlayControlRequest({ action: 'pause' }, 'playing')).toEqual({ action: 'pause' });
    expect(validateAgentPlayControlRequest({ action: 'resume' }, 'paused')).toEqual({ action: 'resume' });
    expect(validateAgentPlayControlRequest({ action: 'stop' }, 'playing')).toEqual({ action: 'stop' });
    expect(validateAgentPlayControlRequest({ action: 'stop' }, 'paused')).toEqual({ action: 'stop' });
  });

  it('defaults a paused step to one fixed tick', () => {
    expect(validateAgentPlayControlRequest({ action: 'step' }, 'paused')).toEqual({
      action: 'step',
      ticks: 1,
    });
  });

  it('accepts an explicit bounded fixed-tick count', () => {
    expect(validateAgentPlayControlRequest({ action: 'step', ticks: 8 }, 'paused')).toEqual({
      action: 'step',
      ticks: 8,
    });
  });

  it('rejects stepping outside a paused Play World', () => {
    expect(() => validateAgentPlayControlRequest({ action: 'step' }, 'playing')).toThrow(
      'Play step requires a paused Play World',
    );
    expect(() => validateAgentPlayControlRequest({ action: 'step' }, 'stopped')).toThrow(
      'Play step requires a paused Play World',
    );
  });

  it('rejects invalid lifecycle transitions', () => {
    expect(() => validateAgentPlayControlRequest({ action: 'play' }, 'playing')).toThrow();
    expect(() => validateAgentPlayControlRequest({ action: 'resume' }, 'playing')).toThrow();
    expect(() => validateAgentPlayControlRequest({ action: 'pause' }, 'paused')).toThrow();
    expect(() => validateAgentPlayControlRequest({ action: 'stop' }, 'stopped')).toThrow();
  });

  it('rejects invalid tick counts and tick payloads on lifecycle actions', () => {
    expect(() => validateAgentPlayControlRequest({ action: 'step', ticks: 0 }, 'paused')).toThrow();
    expect(() => validateAgentPlayControlRequest({ action: 'step', ticks: 1025 }, 'paused')).toThrow();
    expect(() => validateAgentPlayControlRequest({ action: 'pause', ticks: 1 }, 'playing')).toThrow(
      'ticks is only valid for the step action',
    );
  });
});
