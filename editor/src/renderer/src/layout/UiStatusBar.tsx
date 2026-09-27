import { AlertTriangle, Circle, GitBranch } from 'lucide-react';

import type { EditorRuntimeState, StartupState } from '../app/workbenchTypes';
import { useEditorActivityProgress } from '../jobs/editorActivityProgress';
import type { EditorJobProgress } from '../jobs/editorJobProgress';
import { useEditorJobProgress } from '../jobs/editorJobProgress';

import './uiStatusBar.css';

type UiStatusBarProps = {
  startupState: StartupState | null;
  activeScene?: string;
  lastCommand: string;
  aiControl?: string;
  jobProgress?: EditorJobProgress | null;
  runtimeState?: EditorRuntimeState;
  runtimeError?: string;
};

export function UiStatusBar({
  startupState,
  activeScene,
  aiControl,
  jobProgress,
  runtimeState = 'stopped',
  runtimeError,
}: UiStatusBarProps) {
  useEditorActivityProgress(startupState, activeScene, jobProgress === undefined);
  const trackedJobs = useEditorJobProgress();
  const jobs = jobProgress === undefined ? trackedJobs : jobProgress;
  const jobPercent =
    jobs && !jobs.indeterminate && jobs.total > 0 ? Math.round((jobs.completed / jobs.total) * 100) : 0;
  const statusText = jobs
    ? jobs.indeterminate || jobs.total <= 0
      ? jobs.label
      : `${jobs.label} (${jobs.completed} / ${jobs.total})`
    : '';

  return (
    <footer className="status-bar">
      <span>
        <GitBranch size={13} /> main
      </span>
      <span>
        <Circle size={10} /> {startupState?.engineHostConnected ? 'host connected' : 'host unavailable'}
      </span>
      <span className={`status-runtime is-${runtimeState}`} title={runtimeError}>
        {runtimeState === 'faulted' ? <AlertTriangle size={11} /> : <Circle size={9} />}
        {runtimeState === 'stopped'
          ? 'Authoring World'
          : `Play World · ${runtimeState[0].toUpperCase()}${runtimeState.slice(1)}`}
      </span>
      {aiControl && <span className="status-ai-control">{aiControl}</span>}
      <span className="status-spacer" />
      {jobs && (
        <span className="status-jobs" title={statusText}>
          <span className="status-job-label">{statusText}</span>
          <span
            aria-label={
              jobs.indeterminate || jobs.total <= 0
                ? `Editor activity: ${jobs.label}`
                : `${jobs.label}: ${jobs.completed} of ${jobs.total} complete`
            }
            aria-valuemax={jobs.indeterminate || jobs.total <= 0 ? undefined : jobs.total}
            aria-valuemin={jobs.indeterminate || jobs.total <= 0 ? undefined : 0}
            aria-valuenow={jobs.indeterminate || jobs.total <= 0 ? undefined : jobs.completed}
            className={`status-job-progress${jobs.indeterminate ? ' is-indeterminate' : ''}`}
            role="progressbar"
          >
            <span
              className="status-job-progress-fill"
              style={jobs.indeterminate ? undefined : { width: `${jobPercent}%` }}
            />
          </span>
        </span>
      )}
    </footer>
  );
}
