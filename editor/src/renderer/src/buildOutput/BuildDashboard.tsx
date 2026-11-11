import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Activity, Clock3, X } from 'lucide-react';

import type { ArcBuildDiagnostic, ArcBuildRun, ArcBuildSnapshot } from '../../../common/buildTypes';

const formatDuration = (milliseconds: number): string => {
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  if (milliseconds < 60000) return `${(milliseconds / 1000).toFixed(1)} s`;
  return `${Math.floor(milliseconds / 60000)}m ${Math.round((milliseconds % 60000) / 1000)}s`;
};

function BuildDurationChart({ history }: { history: readonly ArcBuildRun[] }) {
  if (!history.length) {
    return <p className="build-dashboard-empty">No completed commands yet. Run a build to populate the graph.</p>;
  }

  const maximum = Math.max(1, ...history.map((run) => run.durationMs));
  const plot = { left: 62, top: 20, width: 650, height: 160 };
  const step = plot.width / history.length;
  const barWidth = Math.min(36, step * 0.62);

  return (
    <svg
      className="build-dashboard-chart"
      viewBox="0 0 730 222"
      role="img"
      aria-label="Recent build command durations, color-coded by result"
      preserveAspectRatio="xMidYMid meet"
    >
      {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
        const y = plot.top + plot.height * (1 - fraction);
        return (
          <g key={fraction}>
            <line x1={plot.left} x2={plot.left + plot.width} y1={y} y2={y} className="build-chart-grid" />
            <text x={plot.left - 9} y={y + 4} textAnchor="end" className="build-chart-axis">
              {formatDuration(maximum * fraction)}
            </text>
          </g>
        );
      })}
      {history.map((run, index) => {
        const barHeight = Math.max(2, (run.durationMs / maximum) * plot.height);
        const x = plot.left + step * (index + 0.5) - barWidth / 2;
        const y = plot.top + plot.height - barHeight;
        return (
          <g key={run.id}>
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              rx={3}
              className={`build-chart-bar is-${run.result}`}
            >
              <title>{`${run.kind} · ${formatDuration(run.durationMs)} · ${run.result} · ${new Date(run.startedAt).toLocaleTimeString()}`}</title>
            </rect>
            <text x={x + barWidth / 2} y={plot.top + plot.height + 18} textAnchor="middle" className="build-chart-axis">
              {run.kind}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function BuildDashboard({
  snapshot,
  onClose,
  onOpenDiagnostic,
}: {
  snapshot: ArcBuildSnapshot | null;
  onClose: () => void;
  onOpenDiagnostic: (diagnostic: ArcBuildDiagnostic) => void;
}) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  const history = snapshot?.history ?? [];
  const successfulBuilds = history.filter((run) => run.kind === 'build' && run.result === 'succeeded');
  const averageMs = successfulBuilds.length
    ? successfulBuilds.reduce((total, run) => total + run.durationMs, 0) / successfulBuilds.length
    : null;
  const failures = history.filter((run) => run.result === 'failed').length;
  const last = history.at(-1);
  const diagnostics = snapshot?.diagnostics ?? [];
  const warnings = diagnostics.filter((item) => item.severity === 'warning').length;
  const errors = diagnostics.filter((item) => item.severity === 'error').length;

  return createPortal(
    <div
      className="build-dashboard-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section role="dialog" aria-modal="true" aria-label="Build dashboard" className="build-dashboard">
        <header className="build-dashboard-heading">
          <div>
            <h2>Build dashboard</h2>
            <p>Project build activity · This editor session</p>
          </div>
          <div className="build-dashboard-heading-actions">
            <span className={`build-dashboard-state is-${snapshot?.state ?? 'idle'}`}>
              <Activity size={14} />
              {snapshot?.state ?? 'idle'}
            </span>
            <button type="button" aria-label="Close build dashboard" onClick={onClose}>
              <X size={17} />
            </button>
          </div>
        </header>

        <div className="build-dashboard-body">
          <div className="build-dashboard-stats">
            <div><small>Commands recorded</small><strong>{history.length}</strong></div>
            <div><small>Latest command</small><strong>{last ? formatDuration(last.durationMs) : '—'}</strong></div>
            <div><small>Avg. successful build</small><strong>{averageMs === null ? '—' : formatDuration(averageMs)}</strong></div>
            <div><small>Failed commands</small><strong>{failures}</strong></div>
            <div><small>Current diagnostics</small><strong>{errors} errors · {warnings} warnings</strong></div>
          </div>

          <section className="build-dashboard-section" aria-label="Build history graph">
            <div className="build-dashboard-section-heading">
              <div>
                <h3>Command duration</h3>
                <p>Measured CMake/tool execution time; not an estimated completion percentage.</p>
              </div>
              <div className="build-dashboard-legend" aria-label="Build results legend">
                <span className="is-succeeded">Succeeded</span>
                <span className="is-failed">Failed</span>
                <span className="is-cancelled">Cancelled</span>
              </div>
            </div>
            <BuildDurationChart history={history.slice(-12)} />
          </section>

          <div className="build-dashboard-details">
            <section className="build-dashboard-section">
              <h3>Recent commands</h3>
              {history.length ? (
                <div className="build-dashboard-table-wrap">
                  <table className="build-dashboard-table">
                    <thead><tr><th>Command</th><th>Configuration</th><th>Duration</th><th>Result</th></tr></thead>
                    <tbody>
                      {[...history].reverse().map((run) => (
                        <tr key={run.id}>
                          <td>{run.kind}</td>
                          <td>{run.configuration}</td>
                          <td>{formatDuration(run.durationMs)}</td>
                          <td className={`is-${run.result}`}>{run.result}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : <p className="build-dashboard-empty">No build history in this session.</p>}
            </section>

            <section className="build-dashboard-section">
              <div className="build-dashboard-section-heading">
                <h3>Latest output</h3>
                <span className="build-dashboard-output-count"><Clock3 size={13} /> {diagnostics.length} lines</span>
              </div>
              <div className="build-dashboard-diagnostics">
                {diagnostics.length ? diagnostics.slice(-100).map((diagnostic) => (
                  <button
                    key={diagnostic.sequence}
                    type="button"
                    disabled={!diagnostic.file}
                    className={`build-dashboard-diagnostic is-${diagnostic.severity}`}
                    title={diagnostic.file ? `${diagnostic.file}:${diagnostic.line ?? 1}` : diagnostic.message}
                    onClick={() => onOpenDiagnostic(diagnostic)}
                  >
                    <span>{diagnostic.severity}</span>
                    <span>{diagnostic.message}</span>
                  </button>
                )) : <p className="build-dashboard-empty">Build output will appear here.</p>}
              </div>
            </section>
          </div>
        </div>
      </section>
    </div>,
    document.body,
  );
}
