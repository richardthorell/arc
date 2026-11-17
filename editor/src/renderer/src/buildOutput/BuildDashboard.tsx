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
  RefreshCw,
  Search,
  X,
  XCircle,
} from 'lucide-react';

import type { ArcBuildDiagnostic, ArcBuildSnapshot } from '../../../common/buildTypes';
import type { BuildHistoryDetail, BuildHistoryPage, BuildRawLogChunk } from '../../../common/buildHistoryTypes';
import type { BuildMonitorEvent, BuildMonitorRun, BuildMonitorSnapshot } from '../../../common/buildMonitorTypes';
import { BuildStageTimeline } from './BuildStageTimeline';
import { BuildTargetGraphPanel } from './BuildTargetGraphPanel';

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
}: {
  snapshot: ArcBuildSnapshot | null;
  onClose: () => void;
  onOpenDiagnostic: (diagnostic: ArcBuildDiagnostic) => void;
  onCancel?: () => void;
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
  const projectRef = useRef<string | null | undefined>(undefined);
  const overviewRequest = useRef(0);
  const selectedRef = useRef<string | null>(null);
  const detailRef = useRef<DetailView | null>(null);
  const detailRequest = useRef(0);
  const reading = useRef(false);

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
      projectRef.current = nextHistory.projectGuid;
      setMonitor(nextMonitor);
      setHistory(nextHistory);
      const available = new Set([...nextMonitor.runs, ...nextHistory.runs].map((run) => run.id));
      setSelectedRunId((current) =>
        !changedProject && current && available.has(current)
          ? current
          : (nextMonitor.activeRunId ?? nextHistory.runs[0]?.id ?? nextMonitor.runs.at(-1)?.id ?? null),
      );
      setError(nextHistory.storageError ?? '');
    } catch (reason) {
      if (request === overviewRequest.current) setError(String(reason));
    } finally {
      if (request === overviewRequest.current) setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

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
      if (event.kind === 'run-started' || event.kind === 'stage-changed' || event.kind === 'run-completed' || event.kind === 'target-graph-changed') {
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

  const visibleTime = selected
    ? (selected.durationMs ?? (active ? Math.max(0, elapsed - Date.parse(selected.startedAt)) : undefined))
    : undefined;
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

  return createPortal(
    <div
      className="build-dashboard-backdrop"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target) onClose();
      }}
    >
      <section className="build-dashboard build-monitor" role="dialog" aria-modal="true" aria-label="Build monitor">
        <header className="build-monitor-header">
          <div className="build-monitor-heading">
            <Activity size={19} aria-hidden="true" />
            <div>
              <h2>Build Monitor</h2>
              <p>Live build activity and project history</p>
            </div>
          </div>
          <div className="build-monitor-actions">
            {active && (selected?.source === 'workflow' || onCancel) && (
              <button type="button" onClick={() => void cancelSelected()} className="build-monitor-cancel">
                <Ban size={14} /> Cancel build
              </button>
            )}
            <button
              type="button"
              onClick={() => void refresh()}
              aria-label="Refresh build history"
              title="Refresh history"
            >
              <RefreshCw size={16} />
            </button>
            <button type="button" onClick={onClose} aria-label="Close build monitor" title="Close">
              <X size={17} />
            </button>
          </div>
        </header>

        <div className="build-monitor-layout">
          <aside className="build-monitor-sidebar" aria-label="Build history">
            <div className="build-monitor-sidebar-title">
              <strong>Build history</strong>
              <span>{history?.total ?? 0}</span>
            </div>
            {activeId && (
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
                      <Status state={run.state} />
                      <strong>
                        {run.action} · {run.configuration}
                      </strong>
                      <span>{run.platform ?? run.targetId ?? 'Project'} · Current</span>
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
                    key={run.id}
                    type="button"
                    className={`build-monitor-history-row${selectedRunId === run.id ? ' is-selected' : ''}`}
                    aria-current={selectedRunId === run.id ? 'true' : undefined}
                    onClick={() => setSelectedRunId(run.id)}
                  >
                    <Status state={run.state} />
                    <strong>
                      {run.action} · {run.configuration}
                    </strong>
                    <span>{time(run.startedAt)}</span>
                    <span>
                      {duration(run.durationMs)} · {run.platform ?? run.source}
                    </span>
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
                <div className="build-monitor-summary">
                  <div>
                    <p className="build-monitor-eyebrow">
                      {selectedRunId === activeId ? 'CURRENT BUILD' : 'RECORDED BUILD'}
                    </p>
                    <h3>
                      {selected.action} <span>· {selected.configuration}</span>
                    </h3>
                    <p>
                      {selected.platform ?? 'Project'}
                      {selected.profile ? ` · ${selected.profile}` : ''}
                      {selected.targetId ? ` · ${selected.targetId}` : ''}
                    </p>
                  </div>
                  <Status state={selected.state} />
                </div>
                <div className="build-monitor-metrics" aria-label="Build summary">
                  <div>
                    <small>Duration</small>
                    <strong>{duration(visibleTime)}</strong>
                  </div>
                  <div>
                    <small>Stages</small>
                    <strong>
                      {selected.stages.filter((stage) => stage.state === 'succeeded').length} / {selected.stages.length}
                    </strong>
                  </div>
                  <div>
                    <small>Errors</small>
                    <strong className="is-error">{selected.errorCount}</strong>
                  </div>
                  <div>
                    <small>Warnings</small>
                    <strong className="is-warning">{selected.warningCount}</strong>
                  </div>
                  <div>
                    <small>Started</small>
                    <strong className="build-monitor-start-time">{time(selected.startedAt)}</strong>
                  </div>
                </div>
                <section className="build-monitor-stages" aria-label="Build stages">
                  <div className="build-monitor-section-title">
                    <h4>Build stages</h4>
                    <span>Elapsed time · actual stage timing, not completion estimates</span>
                  </div>
                  <BuildStageTimeline run={selected} nowMs={elapsed} />
                  {selected.error && (
                    <p role="alert" className="build-monitor-stage-error">
                      {selected.error}
                    </p>
                  )}
                </section>
                <BuildTargetGraphPanel key={selected.id} run={selected} nowMs={elapsed} />
                <section className="build-monitor-output" aria-label="Build logs and diagnostics">
                  <div className="build-monitor-output-header">
                    <div className="build-monitor-tabs" role="tablist" aria-label="Output category">
                      <button
                        role="tab"
                        aria-selected={tab === 'output'}
                        type="button"
                        onClick={() => setTab('output')}
                      >
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
    </div>,
    document.body,
  );
}
