import type { BuildMonitorRun, BuildMonitorStage } from '../../../common/buildMonitorTypes';

const parseTime = (value: string | undefined): number | null => {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
};

export const formatTimelineDuration = (milliseconds: number): string => {
  const value = Math.max(0, Math.round(milliseconds));
  if (value < 1000) return `${value} ms`;
  if (value < 60_000) return `${(value / 1000).toFixed(1)} s`;
  return `${Math.floor(value / 60_000)}m ${Math.floor((value % 60_000) / 1000)}s`;
};

export type TimelineSegment = {
  startMs: number;
  durationMs: number;
  leftPercent: number;
  widthPercent: number;
};

/** All positions derive from recorded times; pending/unmeasured stages are not plotted. */
export function measuredStageTimeline(
  run: BuildMonitorRun,
  nowMs: number,
): { totalMs: number; segments: Map<string, TimelineSegment> } {
  const origin = parseTime(run.startedAt);
  const segments = new Map<string, TimelineSegment>();
  if (origin === null) return { totalMs: 1, segments };

  const runEnd = parseTime(run.completedAt);
  const running = run.state === 'running';
  const runDuration = running
    ? Math.max(0, nowMs - origin)
    : (run.durationMs ?? (runEnd === null ? 0 : Math.max(0, runEnd - origin)));
  let maxExtent = runDuration;
  const measured = new Map<string, { startMs: number; durationMs: number }>();

  for (const stage of run.stages) {
    const start = parseTime(stage.startedAt);
    if (start === null || start < origin - 1000) continue;
    const completed = parseTime(stage.completedAt);
    const end =
      completed ??
      (stage.state === 'running' && running
        ? Math.max(start, nowMs)
        : stage.durationMs !== undefined
          ? start + stage.durationMs
          : null);
    if (end === null || !Number.isFinite(end) || end < start) continue;
    const startMs = Math.max(0, start - origin);
    // A monotonic stage duration is more precise than subtracting rounded ISO timestamps.
    const durationMs =
      stage.state === 'running' && running
        ? Math.max(0, nowMs - start)
        : (stage.durationMs ?? Math.max(0, end - start));
    measured.set(stage.id, { startMs, durationMs });
    maxExtent = Math.max(maxExtent, startMs + durationMs);
  }

  const totalMs = Math.max(1, maxExtent);
  for (const [id, segment] of measured) {
    segments.set(id, {
      ...segment,
      leftPercent: Math.max(0, Math.min(100, (segment.startMs / totalMs) * 100)),
      widthPercent: Math.max(
        0,
        Math.min(100 - (segment.startMs / totalMs) * 100, (segment.durationMs / totalMs) * 100),
      ),
    });
  }
  return { totalMs, segments };
}

const stageDuration = (stage: BuildMonitorStage, segment: TimelineSegment | undefined): string =>
  segment
    ? formatTimelineDuration(segment.durationMs)
    : stage.durationMs !== undefined
      ? formatTimelineDuration(stage.durationMs)
      : '—';

export function BuildStageTimeline({ run, nowMs }: { run: BuildMonitorRun; nowMs: number }) {
  const timeline = measuredStageTimeline(run, nowMs);
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="build-stage-timeline">
      <div className="build-stage-timeline-axis" aria-hidden="true">
        <span>Stage</span>
        <div className="build-stage-timeline-ticks">
          {ticks.map((tick) => (
            <span key={tick} style={{ left: `${tick * 100}%` }}>
              {formatTimelineDuration(tick * timeline.totalMs)}
            </span>
          ))}
        </div>
        <span>Duration</span>
      </div>
      <ol className="build-stage-timeline-rows">
        {run.stages.map((stage) => {
          const segment = timeline.segments.get(stage.id);
          const label = stageDuration(stage, segment);
          return (
            <li key={stage.id} className={`build-stage-timeline-row is-${stage.state}`}>
              <div className="build-stage-timeline-name">
                <span className={`build-stage-timeline-state is-${stage.state}`} aria-label={stage.state} />
                <span title={stage.name}>{stage.name}</span>
              </div>
              <div className="build-stage-timeline-track">
                {ticks.slice(1, -1).map((tick) => (
                  <span key={tick} className="build-stage-timeline-gridline" style={{ left: `${tick * 100}%` }} />
                ))}
                {segment ? (
                  <span
                    role="img"
                    aria-label={`${stage.name}: ${stage.state}; started ${formatTimelineDuration(segment.startMs)} into build; elapsed ${label}`}
                    title={`${stage.name} · ${stage.state} · starts at ${formatTimelineDuration(segment.startMs)} · ${label}`}
                    className={`build-stage-timeline-bar is-${stage.state}`}
                    style={{ left: `${segment.leftPercent}%`, width: `${segment.widthPercent}%` }}
                  />
                ) : (
                  <span className="build-stage-timeline-unmeasured">
                    {stage.state === 'pending' ? 'Waiting' : 'Timing unavailable'}
                  </span>
                )}
              </div>
              <span className="build-stage-timeline-duration">{label}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
