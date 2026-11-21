import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  Activity,
  AlertCircle,
  Ban,
  CheckCircle2,
  ChevronRight,
  Clock3,
  FileText,
  LoaderCircle,
  Search,
  X,
  XCircle,
  CirclePause,
  RotateCcw,
  FolderOpen,
  PackageCheck,
  SlidersHorizontal,
  Trash2,
} from 'lucide-react';

import type { ArcBuildDiagnostic, ArcBuildSnapshot } from '../../../common/buildTypes';
import type { BuildHistoryDetail, BuildHistoryPage, BuildRawLogChunk } from '../../../common/buildHistoryTypes';
import type { BuildMonitorEvent, BuildMonitorRun, BuildMonitorSnapshot } from '../../../common/buildMonitorTypes';
import { BuildExecutionTimeline } from './BuildExecutionTimeline';
import { BuildHistoryComparisonPanel } from './BuildHistoryComparisonPanel';
import { buildDisplayState, buildDuration, buildFailureReason, canRetryBuild } from './buildMonitorPresentation';

const MAX_DISPLAY_CHUNKS = 1200;
const MAX_DISPLAY_EVENTS = 1600;
const isActive = (run: BuildMonitorRun | undefined): boolean => run?.state === 'running';

function duration(milliseconds: number | undefined): string {
  if (milliseconds === undefined || !Number.isFinite(milliseconds)) return '—';
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  if (milliseconds < 60_000) return `${(milliseconds / 1000).toFixed(1)} s`;
  return `${Math.floor(milliseconds / 60_000)}m ${Math.round((milliseconds % 60_000) / 1000)}s`;
}

function time(iso: string | undefined): string {
  if (!iso) return '—';
  const value = new Date(iso);
  return Number.isNaN(value.getTime()) ? '—' : value.toLocaleString();
}

type DetailView = {
  runId: string;
  logs: BuildRawLogChunk[];
  events: BuildMonitorEvent[];
  logCursor: number;
  eventCursor: number;
  moreLogs: boolean;
  moreEvents: boolean;
  hiddenLogChunks: number;
};

function initialDetails(value: BuildHistoryDetail): DetailView {
  return {
    runId: value.run.id,
    logs: value.logs.items,
    events: value.events.items,
    logCursor: value.logs.nextCursor,
    eventCursor: value.events.nextCursor,
    moreLogs: value.logs.hasMore,
    moreEvents: value.events.hasMore,
    hiddenLogChunks: 0,
  };
}

function mergeDetails(current: DetailView, value: BuildHistoryDetail): DetailView {
  const logs = [...current.logs, ...value.logs.items];
  const events = [...current.events, ...value.events.items];
  const hidden = Math.max(0, logs.length - MAX_DISPLAY_CHUNKS);
  return {
    ...current,
    logs: logs.slice(-MAX_DISPLAY_CHUNKS),
    events: events.slice(-MAX_DISPLAY_EVENTS),
    logCursor: value.logs.nextCursor,
    eventCursor: value.events.nextCursor,
    moreLogs: value.logs.hasMore,
    moreEvents: value.events.hasMore,
    hiddenLogChunks: current.hiddenLogChunks + hidden,
  };
}

function Status({ state }: { state: BuildMonitorRun['state'] | 'idle' }) {
  const Icon =
    state === 'running'
      ? LoaderCircle
      : state === 'succeeded'
        ? CheckCircle2
        : state === 'failed'
          ? XCircle
          : state === 'cancelled'
            ? Ban
            : state === 'interrupted'
              ? CirclePause
              : Clock3;
  return (
    <span className={`build-monitor-status is-${state}`}>
      <Icon size={14} aria-hidden="true" />
      {state}
    </span>
  );
}

/** BM3 presents measured stage status only; actual target dependencies belong to BM4/BM5. */
export function BuildDashboard({
  snapshot,
  onClose,
  onOpenDiagnostic,
  onCancel,
  presentation = 'dialog',
}: {
  snapshot: ArcBuildSnapshot | null;
  onClose: () => void;
  onOpenDiagnostic: (diagnostic: ArcBuildDiagnostic) => void;
  onCancel?: () => void;
  presentation?: 'dialog' | 'workspace';
}) {
  const api = window.arc?.build;
  const [monitor, setMonitor] = useState<BuildMonitorSnapshot | null>(null);
  const [history, setHistory] = useState<BuildHistoryPage | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<DetailView | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [detailEpoch, setDetailEpoch] = useState(0);
  const [search, setSearch] = useState('');
  const [tab, setTab] = useState<'output' | 'diagnostics'>('output');
  const [severity, setSeverity] = useState<'all' | 'error' | 'warning'>('all');
  const [stream, setStream] = useState<'all' | 'stdout' | 'stderr'>('all');
  const [elapsed, setElapsed] = useState(Date.now());
  const [retrying, setRetrying] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [clearingHistory, setClearingHistory] = useState(false);
  const projectRef = useRef<string | null | undefined>(undefined);
  const overviewRequest = useRef(0);
  const selectedRef = useRef<string | null>(null);
  const detailRef = useRef<DetailView | null>(null);
  const detailRequest = useRef(0);
  const reading = useRef(false);
  const pendingRunSelection = useRef<string | null>(null);

  const refresh = useCallback(async () => {
    if (!api) {
      setLoading(false);
      return;
    }
    const request = ++overviewRequest.current;
    try {
      const [nextMonitor, nextHistory] = await Promise.all([api.monitorSnapshot(), api.history(0, 100)]);
      if (request !== overviewRequest.current) return;
      const changedProject = projectRef.current !== undefined && projectRef.current !== nextHistory.projectGuid;
      if (changedProject) pendingRunSelection.current = null;
      projectRef.current = nextHistory.projectGuid;
      setMonitor(nextMonitor);
      setHistory(nextHistory);
      const available = new Set([...nextMonitor.runs, ...nextHistory.runs].map((run) => run.id));
      const pending = changedProject ? null : pendingRunSelection.current;
      const resolvedPending = Boolean(pending && available.has(pending));
      setSelectedRunId((current) => {
        if (pending) return pending;
        return !changedProject && current && available.has(current)
          ? current
          : (nextMonitor.activeRunId ?? nextHistory.runs[0]?.id ?? nextMonitor.runs.at(-1)?.id ?? null);
      });
      if (resolvedPending) {
        pendingRunSelection.current = null;
        setDetailEpoch((value) => value + 1);
      }
      setError(nextHistory.storageError ?? '');
    } catch (reason) {
      if (request === overviewRequest.current) setError(String(reason));
    } finally {
      if (request === overviewRequest.current) setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    if (presentation !== 'dialog') return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose, presentation]);

  useEffect(() => {
    const requestRef = overviewRequest;
    let update: ReturnType<typeof setTimeout> | undefined;
    const scheduleRefresh = () => {
      if (update) return;
      update = setTimeout(() => {
        update = undefined;
        void refresh();
      }, 150);
    };
    const unsubscribe = api?.onMonitorEvent((event) => {
      // A newly started build is the user's active work and must become the
      // selected run even when it was initiated outside this view. Later
      // stage/trace events never steal selection from historical inspection.
      if (event.kind === 'run-started') {
        pendingRunSelection.current = event.runId;
        setSelectedRunId(event.runId);
      }
      if (
        event.kind === 'run-started' ||
        event.kind === 'stage-changed' ||
        event.kind === 'run-completed' ||
        event.kind === 'target-graph-changed' ||
        event.kind === 'compile-trace-changed'
      ) {
        scheduleRefresh();
      }
    });
    void refresh();
    return () => {
      requestRef.current++;
      if (update) clearTimeout(update);
      unsubscribe?.();
    };
  }, [api, refresh]);

  useEffect(() => {
    selectedRef.current = selectedRunId;
    const request = ++detailRequest.current;
    detailRef.current = null;
    reading.current = false;
    setDetail(null);
    setLoadingDetail(Boolean(selectedRunId && api));
    if (!selectedRunId || !api) return;
    let cancelled = false;
    void api
      .historyDetails(selectedRunId)
      .then((response) => {
        if (cancelled || request !== detailRequest.current) return;
        const value = response ? initialDetails(response) : null;
        detailRef.current = value;
        setDetail(value);
      })
      .catch((reason: unknown) => {
        if (!cancelled && request === detailRequest.current) setError(String(reason));
      })
      .finally(() => {
        if (!cancelled && request === detailRequest.current) setLoadingDetail(false);
      });
    return () => {
      cancelled = true;
    };
  }, [api, selectedRunId, history?.projectGuid, detailEpoch]);

  const loadNext = useCallback(async () => {
    const current = detailRef.current;
    if (!api || !current || reading.current || selectedRef.current !== current.runId) return;
    reading.current = true;
    setLoadingMore(true);
    const request = detailRequest.current;
    try {
      const response = await api.historyDetails(current.runId, current.logCursor, current.eventCursor);
      if (!response || request !== detailRequest.current || selectedRef.current !== current.runId) return;
      const next = mergeDetails(current, response);
      detailRef.current = next;
      setDetail(next);
    } catch (reason) {
      if (request === detailRequest.current) setError(String(reason));
    } finally {
      if (request === detailRequest.current) {
        reading.current = false;
        setLoadingMore(false);
      }
    }
  }, [api]);

  const runs = history?.runs ?? [];
  const activeId = monitor?.activeRunId ?? runs.find((run) => run.state === 'running')?.id;
  const clearHistory = async () => {
    if (!api?.clearHistory || activeId || clearingHistory || !history?.total) return;
    if (!window.confirm('Clear all build history for this project? This cannot be undone.')) return;
    setClearingHistory(true);
    try {
      await api.clearHistory();
      setSelectedRunId(null);
      setDetail(null);
      await refresh();
    } catch (reason) {
      setError(String(reason));
    } finally {
      setClearingHistory(false);
    }
  };
  const selected =
    monitor?.runs.find((run) => run.id === selectedRunId) ?? runs.find((run) => run.id === selectedRunId) ?? null;
  const active = selectedRunId === activeId && isActive(selected ?? undefined);

  useEffect(() => {
    if (!active) return;
    const interval = setInterval(() => {
      setElapsed(Date.now());
      void loadNext();
    }, 1100);
    return () => clearInterval(interval);
  }, [active, loadNext]);

  const selectedState = selected?.state;
  useEffect(() => {
    if (selectedRunId && selectedState && selectedState !== 'running') void loadNext();
  }, [selectedRunId, selectedState, loadNext]);

  const cancelSelected = async () => {
    if (!selected || !active) return;
    if (selected.source === 'command') {
      onCancel?.();
      return;
    }
    if (!api) return;
    try {
      const workflow = await api.workflowSnapshot();
      if (!workflow?.requestId || workflow.run?.id !== selected.id) throw new Error('Active workflow changed');
      await api.cancelWorkflow(workflow.requestId);
    } catch (reason) {
      setError(String(reason));
    }
  };

  const visibleTime = selected ? buildDuration(selected, elapsed) : undefined;
  const output = (detail?.logs ?? []).filter(
    (chunk) => (stream === 'all' || chunk.stream === stream) && chunk.text.toLowerCase().includes(search.toLowerCase()),
  );
  const diagnostics = (detail?.events ?? [])
    .filter((event): event is Extract<BuildMonitorEvent, { kind: 'diagnostic' }> => event.kind === 'diagnostic')
    .map((event) => event.diagnostic);
  const latestFallback = selectedRunId === snapshot?.runId ? (snapshot?.diagnostics ?? []) : [];
  const shownDiagnostics = (diagnostics.length ? diagnostics : latestFallback).filter(
    (item) =>
      (severity === 'all' || item.severity === severity) &&
      `${item.file ?? ''} ${item.message}`.toLowerCase().includes(search.toLowerCase()),
  );

  const firstError = diagnostics.find((item) => item.severity === 'error');
  const lastStderr = detail?.logs
    .filter((chunk) => chunk.stream === 'stderr')
    .flatMap((chunk) => chunk.text.split(/\r?\n/))
    .filter(Boolean)
    .at(-1);
  const failure = selected ? buildFailureReason(selected, firstError?.message, lastStderr) : null;
  const anyActiveRun = Boolean(activeId);

  const retrySelected = async (run: BuildMonitorRun) => {
    if (!api || anyActiveRun || !canRetryBuild(run) || retrying) return;
    setRetrying(true);
    setError('');
    try {
      if (run.source === 'workflow') {
        if (!run.request) throw new Error('This older workflow has no saved retry settings');
        await api.executeWorkflow(run.request);
      } else {
        const action = run.action;
        if (action !== 'build' && action !== 'configure' && action !== 'clean') return;
        await api.execute({ action, configuration: run.configuration });
      }
      await refresh();
    } catch (reason) {
      await refresh();
      setError(`Retry failed: ${String(reason)}`);
    } finally {
      setRetrying(false);
    }
  };

  const revealArtifact = async (path: string) => {
    if (!api || !selected) return;
    try {
      const result = await api.revealArtifact(selected.id, path);
      if (result) setError(result);
    } catch (reason) {
      setError(String(reason));
    }
  };

  const monitorContent = (
    <section
      className="build-dashboard build-monitor"
      role={presentation === 'dialog' ? 'dialog' : 'region'}
      aria-modal={presentation === 'dialog' ? true : undefined}
      aria-label="Build monitor"
    >
      {presentation === 'dialog' && !selected && (
        <button
          type="button"
          className="build-monitor-floating-close"
          onClick={onClose}
          aria-label="Close build monitor"
        >
          <X size={17} />
        </button>
      )}
      <div className="build-monitor-layout">
        <aside className="build-monitor-sidebar" aria-label="Build history">
          <div className="build-monitor-sidebar-title">
            <strong>Build history</strong>
            <div>
              <span>{history?.total ?? 0}</span>
              <button
                type="button"
                aria-label="Clear build history"
                title={activeId ? 'Stop the active build before clearing history' : 'Clear build history'}
                disabled={Boolean(activeId || clearingHistory || !history?.total)}
                onClick={() => void clearHistory()}
              >
                <Trash2 size={13} aria-hidden="true" />
              </button>
            </div>
          </div>
          {activeId && runs.some((run) => run.id === activeId) && (
            <div className="build-monitor-sidebar-group">
              <small>RUNNING NOW</small>
              {runs
                .filter((run) => run.id === activeId)
                .map((run) => (
                  <button
                    key={run.id}
                    type="button"
                    className={`build-monitor-history-row${selectedRunId === run.id ? ' is-selected' : ''}`}
                    aria-current={selectedRunId === run.id ? 'true' : undefined}
                    onClick={() => setSelectedRunId(run.id)}
                  >
                    <Status state={buildDisplayState(run)} />
                    <strong>
                      {run.action} · {run.configuration}
                    </strong>
                    <span>Current · {duration(buildDuration(run, elapsed))}</span>
                  </button>
                ))}
            </div>
          )}
          <div className="build-monitor-sidebar-group">
            <small>PREVIOUS BUILDS</small>
            {runs
              .filter((run) => run.id !== activeId)
              .map((run) => (
                <button
                  type="button"
                  className={`build-monitor-history-row${selectedRunId === run.id ? ' is-selected' : ''}`}
                  aria-current={selectedRunId === run.id ? 'true' : undefined}
                  onClick={() => setSelectedRunId(run.id)}
                  key={run.id}
                >
                  <Status state={buildDisplayState(run)} />
                  <strong>
                    {run.action} · {run.configuration}
                  </strong>
                  <span>{time(run.startedAt)}</span>
                  <span>{duration(buildDuration(run, elapsed))}</span>
                </button>
              ))}
            {!runs.length && (
              <p className="build-monitor-empty">{loading ? 'Loading history…' : 'No recorded builds yet.'}</p>
            )}
          </div>
        </aside>

        <main className="build-monitor-content">
          {error && (
            <p role="alert" className="build-monitor-error">
              <AlertCircle size={15} /> {error}
            </p>
          )}
          {selected ? (
            <>
              <div className="build-monitor-compact-summary" aria-label="Build summary">
                <div className="build-monitor-compact-title">
                  <strong>{selected.action}</strong>
                  <span>· {selected.configuration}</span>
                  <small>{selected.platform ?? selected.targetId ?? 'Project'}</small>
                </div>
                <div className="build-monitor-compact-facts">
                  <Status state={buildDisplayState(selected)} />
                  <span>{duration(visibleTime)}</span>
                  <span className={selected.errorCount ? 'is-error' : undefined}>{selected.errorCount} errors</span>
                  <span className={selected.warningCount ? 'is-warning' : undefined}>
                    {selected.warningCount} warnings
                  </span>
                </div>
                <div className="build-monitor-compact-actions">
                  {canRetryBuild(selected) && (
                    <button
                      type="button"
                      onClick={() => void retrySelected(selected)}
                      aria-label="Retry selected build"
                      disabled={anyActiveRun || retrying}
                    >
                      <RotateCcw size={14} /> Retry
                    </button>
                  )}
                  {active && (selected.source === 'workflow' || onCancel) && (
                    <button
                      type="button"
                      onClick={() => void cancelSelected()}
                      className="build-monitor-cancel"
                      aria-label="Cancel build"
                    >
                      <Ban size={14} /> Cancel
                    </button>
                  )}
                  {selected.artifacts?.[0] && (
                    <button type="button" onClick={() => void revealArtifact(selected.artifacts![0].path)}>
                      <FolderOpen size={14} /> Output folder
                    </button>
                  )}
                  <button type="button" onClick={() => setDetailsOpen(true)} aria-label="Open build details">
                    <SlidersHorizontal size={14} /> Details
                  </button>
                  {presentation === 'dialog' && (
                    <button type="button" onClick={onClose} aria-label="Close build monitor" title="Close">
                      <X size={16} />
                    </button>
                  )}
                </div>
              </div>
              {failure && (
                <section
                  className={`build-monitor-problem-compact is-${buildDisplayState(selected)}`}
                  aria-label="Build result details"
                >
                  {buildDisplayState(selected) === 'interrupted' ? (
                    <CirclePause size={16} />
                  ) : (
                    <AlertCircle size={16} />
                  )}
                  <strong>{failure}</strong>
                  {selected.errorCount === 0 && buildDisplayState(selected) === 'failed' && (
                    <small>No compiler diagnostics recorded</small>
                  )}
                  <button
                    type="button"
                    onClick={() => {
                      setTab(firstError ? 'diagnostics' : 'output');
                      setSearch('');
                      setStream(firstError ? 'all' : 'stderr');
                      document.querySelector('.build-monitor-output')?.scrollIntoView?.({ block: 'start' });
                    }}
                  >
                    {firstError ? 'View diagnostic' : 'View error output'} <ChevronRight size={14} />
                  </button>
                </section>
              )}
              <BuildExecutionTimeline
                key={selected.id}
                run={selected}
                onFilterOutput={(outputName) => {
                  setSearch(outputName);
                  setStream('all');
                  setTab('output');
                }}
              />
              <section className="build-monitor-output" aria-label="Build logs and diagnostics">
                <div className="build-monitor-output-header">
                  <div className="build-monitor-tabs" role="tablist" aria-label="Output category">
                    <button role="tab" aria-selected={tab === 'output'} type="button" onClick={() => setTab('output')}>
                      <FileText size={14} /> Output
                    </button>
                    <button
                      role="tab"
                      aria-selected={tab === 'diagnostics'}
                      type="button"
                      onClick={() => setTab('diagnostics')}
                    >
                      <AlertCircle size={14} /> Diagnostics
                    </button>
                  </div>
                  <label className="build-monitor-search">
                    <Search size={14} aria-hidden="true" />
                    <input
                      value={search}
                      onChange={(event) => setSearch(event.target.value)}
                      aria-label="Search build output"
                      placeholder="Search loaded output"
                    />
                  </label>
                </div>
                <div className="build-monitor-output-filters">
                  {tab === 'output' ? (
                    <>
                      <label htmlFor="build-monitor-stream">Stream</label>
                      <select
                        id="build-monitor-stream"
                        value={stream}
                        onChange={(event) => setStream(event.target.value as typeof stream)}
                      >
                        <option value="all">All output</option>
                        <option value="stdout">stdout</option>
                        <option value="stderr">stderr</option>
                      </select>
                    </>
                  ) : (
                    <>
                      <label htmlFor="build-monitor-severity">Severity</label>
                      <select
                        id="build-monitor-severity"
                        value={severity}
                        onChange={(event) => setSeverity(event.target.value as typeof severity)}
                      >
                        <option value="all">All diagnostics</option>
                        <option value="error">Errors</option>
                        <option value="warning">Warnings</option>
                      </select>
                    </>
                  )}
                  <span>
                    {tab === 'output' ? `${output.length} loaded chunks` : `${shownDiagnostics.length} diagnostics`}
                  </span>
                </div>
                <div
                  className="build-monitor-output-scroll"
                  role="tabpanel"
                  aria-label={tab === 'output' ? 'Build output' : 'Build diagnostics'}
                >
                  {tab === 'output' ? (
                    output.length ? (
                      output.map((chunk, index) => (
                        <div key={index} className={`build-monitor-log-chunk is-${chunk.stream}`}>
                          <span className="build-monitor-log-stream" title={time(chunk.timestamp)}>
                            {chunk.stream}
                          </span>
                          <pre>{chunk.text}</pre>
                        </div>
                      ))
                    ) : (
                      <p className="build-monitor-empty">
                        {loadingDetail ? 'Loading output…' : 'No matching output recorded.'}
                      </p>
                    )
                  ) : shownDiagnostics.length ? (
                    shownDiagnostics.map((diagnostic, index) => (
                      <button
                        type="button"
                        key={`${diagnostic.sequence}-${index}`}
                        className={`build-monitor-diagnostic is-${diagnostic.severity}`}
                        onClick={() => onOpenDiagnostic(diagnostic)}
                        disabled={!diagnostic.file}
                        title={diagnostic.file ? `${diagnostic.file}:${diagnostic.line ?? 1}` : diagnostic.message}
                      >
                        <span>{diagnostic.severity}</span>
                        <span>{diagnostic.file ? `${diagnostic.file}:${diagnostic.line ?? 1}` : 'Build'}</span>
                        <span>{diagnostic.message}</span>
                        {diagnostic.file && <ChevronRight size={14} />}
                      </button>
                    ))
                  ) : (
                    <p className="build-monitor-empty">
                      {loadingDetail ? 'Loading diagnostics…' : 'No matching diagnostics recorded.'}
                    </p>
                  )}
                </div>
                <footer className="build-monitor-output-footer">
                  <span>
                    {detail?.hiddenLogChunks
                      ? `${detail.hiddenLogChunks} earlier chunks removed from memory · reopen run to read from start`
                      : 'Full output remains in project build history'}
                  </span>
                  {Boolean(detail?.hiddenLogChunks) && (
                    <button type="button" onClick={() => setDetailEpoch((value) => value + 1)}>
                      Read from start
                    </button>
                  )}
                  {(detail?.moreLogs || detail?.moreEvents) && (
                    <button type="button" disabled={loadingMore} onClick={() => void loadNext()}>
                      {loadingMore ? 'Loading…' : 'Load more'}
                    </button>
                  )}
                </footer>
              </section>
              {detailsOpen && (
                <aside className="build-monitor-details" aria-label="Build details">
                  <header>
                    <strong>Build details</strong>
                    <button type="button" onClick={() => setDetailsOpen(false)} aria-label="Close build details">
                      <X size={16} />
                    </button>
                  </header>
                  <div>
                    <dl>
                      <dt>Started</dt>
                      <dd>{time(selected.startedAt)}</dd>
                      <dt>Profile</dt>
                      <dd>{selected.profile ?? 'Default'}</dd>
                      <dt>Target</dt>
                      <dd>{selected.targetId ?? selected.platform ?? 'Project'}</dd>
                    </dl>
                    {Boolean(selected.artifacts?.length) && (
                      <section className="build-monitor-artifacts" aria-label="Build artifacts">
                        <div className="build-monitor-section-title">
                          <h4>
                            <PackageCheck size={15} /> Build artifacts
                          </h4>
                          <span>{selected.artifacts?.length} published</span>
                        </div>
                        {selected.artifacts?.map((artifact, index) => (
                          <div className="build-monitor-artifact" key={`${artifact.kind}-${artifact.path}-${index}`}>
                            <div>
                              <strong>{artifact.kind}</strong>
                              <code title={artifact.path}>{artifact.path}</code>
                            </div>
                            <button
                              type="button"
                              aria-label={`Reveal ${artifact.kind} artifact`}
                              onClick={() => void revealArtifact(artifact.path)}
                            >
                              <FolderOpen size={14} /> Reveal
                            </button>
                          </div>
                        ))}
                      </section>
                    )}
                    {selected.state === 'succeeded' && (
                      <BuildHistoryComparisonPanel key={selected.id} current={selected} historyRuns={runs} />
                    )}
                  </div>
                </aside>
              )}
            </>
          ) : (
            <div className="build-monitor-welcome">
              <Activity size={30} />
              <h3>No build selected</h3>
              <p>
                {loading
                  ? 'Loading project build history…'
                  : 'Start a build to see its stages, diagnostics and full output here.'}
              </p>
            </div>
          )}
        </main>
      </div>
    </section>
  );

  if (presentation === 'workspace') return <div className="build-monitor-workspace">{monitorContent}</div>;

  return createPortal(
    <div
      className="build-dashboard-backdrop"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target) onClose();
      }}
    >
      {monitorContent}
    </div>,
    document.body,
  );
}
