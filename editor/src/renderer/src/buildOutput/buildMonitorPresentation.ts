import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';

export const buildDuration = (run: BuildMonitorRun, nowMs: number): number | undefined => {
  // A persisted interrupted build has no trustworthy process-end timestamp.
  // Never count idle time between editor sessions as compile duration.
  if (run.state === 'interrupted' || run.error?.startsWith('Interrupted:')) return undefined;
  if (run.durationMs !== undefined) return Number.isFinite(run.durationMs) ? run.durationMs : undefined;
  if (run.state !== 'running') return undefined;
  const started = Date.parse(run.startedAt);
  return Number.isFinite(started) ? Math.max(0, nowMs - started) : undefined;
};

export const canRetryBuild = (run: BuildMonitorRun): boolean =>
  run.state !== 'running' &&
  (run.source === 'workflow'
    ? Boolean(run.request)
    : run.action === 'build' || run.action === 'configure' || run.action === 'clean');

export const buildDisplayState = (run: BuildMonitorRun): BuildMonitorRun['state'] =>
  run.state === 'failed' && run.error?.startsWith('Interrupted:') ? 'interrupted' : run.state;

/** Do not imply that a process exit is the same thing as a compiler diagnostic. */
export const buildFailureReason = (
  run: BuildMonitorRun,
  firstDiagnostic?: string,
  lastStderr?: string,
): string | null => {
  const state = buildDisplayState(run);
  if (state === 'interrupted') return 'The editor closed before the build completed. The finish time is unknown.';
  if (state === 'cancelled') return 'Build cancelled before completion.';
  if (state !== 'failed') return null;
  return (
    run.error ||
    firstDiagnostic ||
    lastStderr ||
    'The build command failed without a recorded compiler diagnostic. Inspect the output for the underlying tool error.'
  );
};

export const availableBuildPlatforms = (runs: BuildMonitorRun[]): string[] =>
  [...new Set(runs.map((run) => run.platform ?? 'Project'))].sort();

export const filterBuildRuns = (
  runs: BuildMonitorRun[],
  status: BuildMonitorRun['state'] | 'all',
  platform: string,
  search: string,
): BuildMonitorRun[] => {
  const term = search.trim().toLowerCase();
  return runs.filter((run) => {
    if (status !== 'all' && buildDisplayState(run) !== status) return false;
    if (platform !== 'all' && (run.platform ?? 'Project') !== platform) return false;
    if (!term) return true;
    return [run.action, run.configuration, run.profile, run.targetId, run.platform, run.error, run.id].some((value) =>
      value?.toLowerCase().includes(term),
    );
  });
};
