import { describe, expect, it } from 'vitest';
import { createStandaloneLaunchRequest, describeStandaloneLaunchFailure } from './standaloneLaunch';

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
