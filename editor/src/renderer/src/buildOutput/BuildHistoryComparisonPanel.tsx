import { useState } from 'react';
import { GitCompareArrows } from 'lucide-react';

import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import { compareBuildHistory, comparisonBaselines, type BuildTimingDelta } from './buildHistoryComparison';

function timeMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function changeLabel(value: BuildTimingDelta): string {
  const sign = value.changeMs > 0 ? '+' : value.changeMs < 0 ? '−' : '';
  const percentage =
    value.changePercent === undefined
      ? 'percentage unavailable (zero baseline)'
      : `${value.changePercent > 0 ? '+' : ''}${value.changePercent.toFixed(1)}%`;
  return `${sign}${timeMs(Math.abs(value.changeMs))} (${percentage})`;
}

/**
 * Archived history comparisons are measurements, not normalized benchmarks.
 * The dropdown is scoped to comparable runs and defaults to the newest
 * successful baseline older than the selected build.
 */
export function BuildHistoryComparisonPanel({
  current,
  historyRuns,
}: {
  current: BuildMonitorRun;
  historyRuns: BuildMonitorRun[];
}) {
  const [choice, setChoice] = useState('latest');
  const candidates = comparisonBaselines(current, historyRuns);
  const baseline = choice === 'latest' ? candidates[0] : candidates.find((run) => run.id === choice);
  const value = baseline ? compareBuildHistory(current, baseline) : null;
  return (
    <section className="build-monitor-comparison" aria-label="Build performance comparison">
      <div className="build-monitor-section-title">
        <h4>
          <GitCompareArrows size={15} aria-hidden="true" /> Build comparison
        </h4>
        <span>Measured history · no estimates</span>
      </div>
      {candidates.length ? (
        <>
          <label className="build-monitor-comparison-picker">
            Compare with
            <select
              aria-label="Comparison baseline"
              value={choice}
              onChange={(event) => setChoice(event.target.value)}
            >
              <option value="latest">Latest comparable successful build</option>
              {candidates.map((run) => (
                <option value={run.id} key={run.id}>
                  {new Date(run.startedAt).toLocaleString()} · {timeMs(run.durationMs!)} · {run.id.slice(0, 8)}
                </option>
              ))}
            </select>
          </label>
          {value && baseline && (
            <>
              <div className="build-monitor-comparison-total">
                <div>
                  <small>Previous</small>
                  <strong>{timeMs(value.total.beforeMs)}</strong>
                </div>
                <div>
                  <small>Selected</small>
                  <strong>{timeMs(value.total.afterMs)}</strong>
                </div>
                <div>
                  <small>Elapsed change</small>
                  <strong>{changeLabel(value.total)}</strong>
                </div>
              </div>
              {value.stages.length > 0 && (
                <div className="build-monitor-comparison-table-wrap">
                  <h5>Measured stage changes</h5>
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">Stage</th>
                        <th scope="col">Previous</th>
                        <th scope="col">Selected</th>
                        <th scope="col">Change</th>
                      </tr>
                    </thead>
                    <tbody>
                      {value.stages.map((stage) => (
                        <tr key={stage.name}>
                          <th scope="row">{stage.name}</th>
                          <td>{timeMs(stage.beforeMs)}</td>
                          <td>{timeMs(stage.afterMs)}</td>
                          <td>{changeLabel(stage)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {current.compileTrace?.source === 'ninja-log' && baseline.compileTrace?.source === 'ninja-log' && (
                <div className="build-monitor-comparison-table-wrap">
                  <h5>Matching Ninja object tasks ({value.matchedObjects})</h5>
                  {value.objects.length > 0 ? (
                    <table>
                      <thead>
                        <tr>
                          <th scope="col">Source / object</th>
                          <th scope="col">Previous</th>
                          <th scope="col">Selected</th>
                          <th scope="col">Change</th>
                        </tr>
                      </thead>
                      <tbody>
                        {value.objects.map((task) => (
                          <tr key={task.output}>
                            <th scope="row" title={task.output}>
                              {task.sourceFile ?? task.output}
                            </th>
                            <td>{timeMs(task.beforeMs)}</td>
                            <td>{timeMs(task.afterMs)}</td>
                            <td>{changeLabel(task)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <p className="build-monitor-target-note">No matching measured object outputs between these runs.</p>
                  )}
                  {value.matchedObjects > value.objects.length && (
                    <p className="build-monitor-target-note">Showing the eight largest increases in task duration.</p>
                  )}
                </div>
              )}
              {current.compileTrace?.source === 'msbuild-file-log' && (
                <p className="build-monitor-target-note">
                  Per-task MSBuild comparisons are omitted: CL task IDs and source grouping are not stable across builds.
                </p>
              )}
            </>
          )}
        </>
      ) : (
        <p className="build-monitor-empty">
          No earlier successful build matches this action, configuration, target and profile.
        </p>
      )}
      <p className="build-monitor-target-note">
        Differences may reflect code changes, incremental caching or machine load. These observations do not establish a
        statistically significant performance regression.
      </p>
    </section>
  );
}
