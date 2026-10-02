import { describe, expect, it } from 'vitest';
import {
  createStandaloneLaunchRequest,
  createStandaloneRelaunchRequest,
  createStandaloneSession,
  describeStandaloneLaunchFailure,
  reduceStandaloneSession,
} from './standaloneLaunch';

describe('standalone launch contract', () => {
  it('creates a versioned request with stable session identity and build configuration', () => {
    expect(createStandaloneLaunchRequest(' session-42 ', ' C:\\projects\\arc-game ', 'development')).toEqual({
      version: 1,
      sessionId: 'session-42',
      projectPath: 'C:/projects/arc-game',
      configuration: 'development',
    });
  });

  it('keeps separate launch requests independently identifiable', () => {
    const first = createStandaloneLaunchRequest('standalone-a', '/projects/game', 'debug');
    const second = createStandaloneLaunchRequest('standalone-b', '/projects/game', 'release');

    expect(first.sessionId).not.toBe(second.sessionId);
    expect(first.configuration).toBe('debug');
    expect(second.configuration).toBe('release');
  });

  it('rejects requests that cannot identify a session or project', () => {
    expect(() => createStandaloneLaunchRequest('   ', '/projects/game', 'development')).toThrow();
    expect(() => createStandaloneLaunchRequest('standalone-a', '   ', 'development')).toThrow();
  });

  it('tracks launch, stop, and stopped states for the identified process', () => {
    const request = createStandaloneLaunchRequest('standalone-a', '/projects/game', 'development');
    const launching = createStandaloneSession(request);
    const running = reduceStandaloneSession(launching, { type: 'started', sessionId: 'standalone-a', processId: 42 });
    const stopping = reduceStandaloneSession(running, { type: 'stop-requested', sessionId: 'standalone-a' });
    const stopped = reduceStandaloneSession(stopping, { type: 'stopped', sessionId: 'standalone-a' });

    expect(running).toMatchObject({ status: 'running', processId: 42 });
    expect(stopping.status).toBe('stopping');
    expect(stopped).toMatchObject({ status: 'stopped' });
    expect(stopped.processId).toBeUndefined();
  });

  it('ignores stale lifecycle events from another standalone session', () => {
    const request = createStandaloneLaunchRequest('current', '/projects/game', 'development');
    const session = createStandaloneSession(request);

    expect(reduceStandaloneSession(session, { type: 'started', sessionId: 'old', processId: 7 })).toBe(session);
  });

  it('relaunches with a new identity while preserving project and configuration', () => {
    const request = createStandaloneLaunchRequest('old', '/projects/game', 'release');
    const session = reduceStandaloneSession(createStandaloneSession(request), {
      type: 'stopped',
      sessionId: 'old',
    });

    expect(createStandaloneRelaunchRequest(session, 'new')).toEqual({
      version: 1,
      sessionId: 'new',
      projectPath: '/projects/game',
      configuration: 'release',
    });
  });

  it('records actionable launch failure state without a process identity', () => {
    const request = createStandaloneLaunchRequest('standalone-a', '/projects/game', 'debug');
    const failed = reduceStandaloneSession(createStandaloneSession(request), {
      type: 'failed',
      sessionId: 'standalone-a',
      failure: { code: 'runtime-missing', message: 'Runtime executable is missing.' },
    });

    expect(failed.status).toBe('failed');
    expect(failed.processId).toBeUndefined();
    expect(failed.failure?.code).toBe('runtime-missing');
  });

  it('preserves actionable host failure detail when available', () => {
    expect(
      describeStandaloneLaunchFailure({
        code: 'runtime-missing',
        message: 'Runtime executable was not found at bin/game.',
      }),
    ).toBe('Runtime executable was not found at bin/game.');
  });

  it('provides actionable fallback messages for known launch failures', () => {
    expect(describeStandaloneLaunchFailure({ code: 'build-required', message: '' })).toContain('Build the project');
    expect(describeStandaloneLaunchFailure({ code: 'launch-failed', message: '' })).toContain('failed to start');
  });
});
