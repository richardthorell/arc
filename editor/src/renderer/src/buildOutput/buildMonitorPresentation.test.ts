import { describe, expect, it } from 'vitest';

import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import {
  availableBuildPlatforms,
  buildDisplayState,
  buildDuration,
  buildFailureReason,
  canRetryBuild,
  filterBuildRuns,
} from './buildMonitorPresentation';

const base: BuildMonitorRun = {
  id: 'build-1',
  projectGuid: 'game',
  source: 'command',
  action: 'build',
  configuration: 'RelWithDebInfo',
  platform: 'windows',
  startedAt: '2026-11-18T12:00:00.000Z',
  state: 'failed',
  completedAt: '2026-11-18T12:00:05.000Z',
  durationMs: 5000,
  stages: [{ id: 'build-1:compile', name: 'Compile', state: 'failed', durationMs: 5000 }],
  errorCount: 0,
  warningCount: 0,
};

describe('Build Monitor actionable presentation', () => {
  it('does not show invented restart-time duration for current or legacy interruptions', () => {
    const interrupted = {
      ...base,
      state: 'interrupted' as const,
      error: 'Editor closed before the build completed',
      completedAt: undefined,
      durationMs: undefined,
    };
    expect(buildDuration(interrupted, Date.parse('2026-12-01T00:00:00Z'))).toBeUndefined();
    expect(buildFailureReason(interrupted)).toMatch(/finish time is unknown/);
    const legacy = {
      ...base,
      error: 'Interrupted: editor closed before build completion',
      durationMs: 1401 * 60_000,
    };
    expect(buildDisplayState(legacy)).toBe('interrupted');
    expect(buildDuration(legacy, Date.now())).toBeUndefined();
  });

  it('distinguishes process failures from compiler diagnostics and supports a useful fallback', () => {
    expect(buildFailureReason(base)).toContain('without a recorded compiler diagnostic');
    expect(buildFailureReason(base, 'linker could not find entry point')).toBe('linker could not find entry point');
    expect(buildFailureReason({ ...base, error: 'CMake configuration failed' }, 'linker error')).toBe(
      'CMake configuration failed',
    );
    expect(buildFailureReason({ ...base, state: 'succeeded' })).toBeNull();
    expect(buildDuration({ ...base, state: 'running', durationMs: undefined }, Date.parse(base.startedAt) + 600)).toBe(
      600,
    );
  });

  it('only allows retry for executable recorded operations', () => {
    expect(canRetryBuild(base)).toBe(true);
    expect(canRetryBuild({ ...base, state: 'running' })).toBe(false);
    expect(canRetryBuild({ ...base, source: 'workflow' })).toBe(false);
    expect(
      canRetryBuild({
        ...base,
        source: 'workflow',
        request: { action: 'build', profile: 'windows', targetId: 'windows', destination: 'none', sessionId: '' },
      }),
    ).toBe(true);
    expect(canRetryBuild({ ...base, action: 'openIde' })).toBe(false);
  });

  it('filters persisted builds by status, platform and search without altering their order', () => {
    const android = {
      ...base,
      id: 'build-2',
      platform: 'android',
      targetId: 'android-arm64-vulkan',
      configuration: 'Debug',
      state: 'succeeded' as const,
    };
    expect(availableBuildPlatforms([base, android])).toEqual(['android', 'windows']);
    expect(filterBuildRuns([base, android], 'failed', 'all', '')).toEqual([base]);
    expect(filterBuildRuns([base, android], 'all', 'android', '')).toEqual([android]);
    expect(filterBuildRuns([base, android], 'all', 'all', 'arm64')).toEqual([android]);
    expect(filterBuildRuns([base, android], 'all', 'all', 'nonexistent')).toEqual([]);
  });
});
