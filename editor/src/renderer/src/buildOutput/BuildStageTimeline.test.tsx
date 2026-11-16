// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import { BuildStageTimeline, measuredStageTimeline } from './BuildStageTimeline';

const origin = Date.parse('2026-11-16T10:00:00.000Z');
const run = (): BuildMonitorRun => ({
  id: 'timeline-run', projectGuid: 'project', source: 'workflow', action: 'build',
  configuration: 'Debug', state: 'succeeded', startedAt: new Date(origin).toISOString(),
  completedAt: new Date(origin + 10000).toISOString(), durationMs: 10000,
  stages: [
    { id: 'validate', name: 'validate', state: 'succeeded', startedAt: new Date(origin).toISOString(), completedAt: new Date(origin + 2000).toISOString(), durationMs: 2000 },
    { id: 'compile', name: 'compile', state: 'succeeded', startedAt: new Date(origin + 2000).toISOString(), completedAt: new Date(origin + 9000).toISOString(), durationMs: 7000 },
    { id: 'package', name: 'package', state: 'succeeded', startedAt: new Date(origin + 9000).toISOString(), completedAt: new Date(origin + 10000).toISOString(), durationMs: 1000 },
  ], errorCount: 0, warningCount: 0,
});

afterEach(cleanup);

describe('BM4 measured stage timeline', () => {
  it('plots recorded relative offsets rather than equally spaced stage order', () => {
    const timeline = measuredStageTimeline(run(), origin + 10000);
    expect(timeline.totalMs).toBe(10000);
    expect(timeline.segments.get('validate')).toMatchObject({ leftPercent: 0, widthPercent: 20 });
    expect(timeline.segments.get('compile')).toMatchObject({ leftPercent: 20, widthPercent: 70 });
    expect(timeline.segments.get('package')).toMatchObject({ leftPercent: 90, widthPercent: 10 });
    render(<BuildStageTimeline run={run()} nowMs={origin + 10000} />);
    expect(screen.getByRole('img', { name: /compile: succeeded; started 2.0 s into build; elapsed 7.0 s/ })).toHaveStyle({ left: '20%', width: '70%' });
  });

  it('extends only the running stage to the current measured time', () => {
    const active: BuildMonitorRun = {
      ...run(), state: 'running', completedAt: undefined, durationMs: undefined,
      stages: [
        { id: 'validate', name: 'validate', state: 'succeeded', startedAt: new Date(origin).toISOString(), completedAt: new Date(origin + 1000).toISOString(), durationMs: 1000 },
        { id: 'compile', name: 'compile', state: 'running', startedAt: new Date(origin + 1000).toISOString() },
        { id: 'package', name: 'package', state: 'pending' },
      ],
    };
    const timeline = measuredStageTimeline(active, origin + 5000);
    expect(timeline.totalMs).toBe(5000);
    expect(timeline.segments.get('compile')).toMatchObject({ startMs: 1000, durationMs: 4000, leftPercent: 20, widthPercent: 80 });
    expect(timeline.segments.has('package')).toBe(false);
    render(<BuildStageTimeline run={active} nowMs={origin + 5000} />);
    expect(screen.getByText('Waiting')).toBeInTheDocument();
  });

  it('does not invent bars for older records with missing stage timestamps', () => {
    const old: BuildMonitorRun = { ...run(), stages: [{ id: 'legacy', name: 'compile', state: 'succeeded', durationMs: 2000 }] };
    expect(measuredStageTimeline(old, origin + 10000).segments.size).toBe(0);
    render(<BuildStageTimeline run={old} nowMs={origin + 10000} />);
    expect(screen.getByText('Timing unavailable')).toBeInTheDocument();
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(screen.getByText('2.0 s')).toBeInTheDocument();
  });

  it('never extends an interrupted historical run until the present', () => {
    const interrupted: BuildMonitorRun = {
      ...run(), state: 'failed', completedAt: new Date(origin + 3000).toISOString(), durationMs: 3000,
      stages: [{ id: 'compile', name: 'compile', state: 'failed', startedAt: new Date(origin).toISOString() }],
    };
    expect(measuredStageTimeline(interrupted, origin + 100000).segments.size).toBe(0);
  });
});
