import { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, Ban, CheckCircle2, ChevronUp, LoaderCircle, XCircle } from 'lucide-react';

import type { BuildMonitorRun, BuildMonitorSnapshot } from '../../../common/buildMonitorTypes';

import './buildActivityTray.css';

export type BuildActivityTrayProps = {
  projectKey: string;
  onOpenBuilds: () => void;
};

const describeRun = (run: BuildMonitorRun | null) => {
  if (!run) return 'No builds yet';
  return `${run.platform ?? 'Project'} · ${run.action} · ${run.configuration}`;
};

/**
 * The tray only observes the main-process build monitor. It never owns the
 * child process: closing this popover or the Builds tab cannot cancel a build.
 */
export function BuildActivityTray({ projectKey, onOpenBuilds }: BuildActivityTrayProps) {
  const [monitor, setMonitor] = useState<BuildMonitorSnapshot | null>(null);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState('');
  const root = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await window.arc?.build?.monitorSnapshot();
      if (next) setMonitor(next);
    } catch (reason) {
      setError(String(reason));
    }
  }, []);

  useEffect(() => {
    // Project switches reset the tray rather than presenting stale build IDs.
    setMonitor(null);
    setOpen(false);
    setError('');
    let alive = true;
    let revision = 0;
    const read = async () => {
      const request = ++revision;
      try {
        const next = await window.arc?.build?.monitorSnapshot();
        if (alive && request === revision) setMonitor(next ?? null);
      } catch (reason) {
        if (alive && request === revision) setError(String(reason));
      }
    };
    const unsubscribeMonitor = window.arc?.build?.onMonitorEvent?.((event) => {
      if (
        event.kind === 'run-started' ||
        event.kind === 'stage-changed' ||
        event.kind === 'run-completed' ||
        event.kind === 'target-graph-changed'
      )
        void read();
    });
    const unsubscribeBuild = window.arc?.build?.onState?.(() => void read());
    void read();
    return () => {
      alive = false;
      revision++;
      unsubscribeMonitor?.();
      unsubscribeBuild?.();
    };
  }, [projectKey]);

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('pointerdown', closeOutside);
    return () => document.removeEventListener('pointerdown', closeOutside);
  }, [open]);

  const activeRun = monitor?.runs.find((run) => run.id === monitor.activeRunId && run.state === 'running');
  const recentRun = activeRun ?? monitor?.runs.at(-1) ?? null;
  const state = activeRun?.state ?? recentRun?.state ?? 'idle';
  const Icon =
    state === 'running' ? LoaderCircle : state === 'failed' ? XCircle : state === 'succeeded' ? CheckCircle2 : Activity;
  const runningStage = activeRun?.stages.find((stage) => stage.state === 'running')?.name;
  const label =
    state === 'running'
      ? `Building${runningStage ? ` · ${runningStage}` : ''}`
      : state === 'idle'
        ? 'Builds'
        : `Build ${state}`;

  const cancel = async () => {
    if (!activeRun) return;
    try {
      const api = window.arc?.build;
      if (!api) return;
      if (activeRun.source === 'workflow') {
        const current = await api.workflowSnapshot();
        if (!current?.requestId || current.run?.id !== activeRun.id) {
          throw new Error('Active build changed; open Builds to review its status.');
        }
        await api.cancelWorkflow(current.requestId);
      } else {
        await api.execute({ action: 'cancel' });
      }
      await refresh();
    } catch (reason) {
      setError(String(reason));
    }
  };

  return (
    <div className="build-activity-tray" ref={root}>
      <button
        type="button"
        className={`build-activity-trigger is-${state}`}
        aria-label={`Build activity: ${label}`}
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => setOpen((value) => !value)}
        title="Build activity"
      >
        <Icon aria-hidden="true" className={state === 'running' ? 'build-activity-spin' : ''} size={14} />
        <span>{label}</span>
        <ChevronUp aria-hidden="true" size={12} />
      </button>
      {open && (
        <section className="build-activity-popover" role="dialog" aria-label="Build activity">
          <header>
            <strong>Build activity</strong>
            <span>{activeRun ? 'Running in background' : 'Latest build'}</span>
          </header>
          <p className="build-activity-project">{describeRun(recentRun)}</p>
          <p className={`build-activity-result is-${state}`}>
            {activeRun
              ? `Running${runningStage ? ` · ${runningStage}` : ''}`
              : recentRun
                ? `${recentRun.state} · ${recentRun.errorCount} errors · ${recentRun.warningCount} warnings`
                : 'No build recorded for this project'}
          </p>
          {error && (
            <p className="build-activity-error" role="alert">
              {error}
            </p>
          )}
          <footer>
            <button
              type="button"
              className="build-activity-open"
              onClick={() => {
                setOpen(false);
                onOpenBuilds();
              }}
            >
              Open Builds
            </button>
            {activeRun && (
              <button type="button" className="build-activity-cancel" onClick={() => void cancel()}>
                <Ban size={13} aria-hidden="true" /> Cancel
              </button>
            )}
          </footer>
        </section>
      )}
    </div>
  );
}
