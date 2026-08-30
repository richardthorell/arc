import { Circle, GitBranch } from 'lucide-react';

import type { StartupState } from '../app/workbenchTypes';
import type { EditorJobProgress } from '../jobs/editorJobProgress';
import { useEditorJobProgress } from '../jobs/editorJobProgress';

import './uiStatusBar.css';

type UiStatusBarProps = {
  startupState: StartupState | null;
  activeScene?: string;
  lastCommand: string;
  aiControl?: string;
  jobProgress?: EditorJobProgress | null;
};

export function UiStatusBar({ startupState, aiControl, jobProgress }: UiStatusBarProps) {
  const trackedJobs = useEditorJobProgress();
  const jobs = jobProgress === undefined ? trackedJobs : jobProgress;
  const jobPercent = jobs && jobs.total > 0 ? Math.round((jobs.completed / jobs.total) * 100) : 0;

  return (
    <footer className="status-bar">
      <span>
        <GitBranch size={13} /> main
      </span>
      <span>
        <Circle size={10} /> {startupState?.engineHostConnected ? 'host connected' : 'host unavailable'}
      </span>
      {aiControl && <span className="status-ai-control">{aiControl}</span>}
      <span className="status-spacer" />
      {jobs && (
        <span className="status-jobs">
          <span>
            Job ({jobs.completed} / {jobs.total})
          </span>
          <span
            aria-label={`Editor jobs: ${jobs.completed} of ${jobs.total} complete`}
            aria-valuemax={jobs.total}
            aria-valuemin={0}
            aria-valuenow={jobs.completed}
            className="status-job-progress"
            role="progressbar"
          >
            <span className="status-job-progress-fill" style={{ width: `${jobPercent}%` }} />
          </span>
        </span>
      )}
    </footer>
  );
}
