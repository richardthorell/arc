import { useEffect, useState } from 'react';
import type { LivePlayDiagnosticsSnapshot } from '../../../common/livePlayDiagnosticsTypes';
import type { WindowsLivePlaySnapshot } from '../../../common/livePlaySessionTypes';
import './livePlayDiagnostics.css';

function sameSession(diagnostics: LivePlayDiagnosticsSnapshot, session: WindowsLivePlaySnapshot): boolean {
  return Boolean(
    session.sessionId &&
    session.instanceId &&
    diagnostics.sessionId === session.sessionId &&
    diagnostics.instanceId === session.instanceId,
  );
}

/** Rehydrate after a renderer reload, but do not overwrite a newer subscribed update. */
export function useLivePlayDiagnostics(session: WindowsLivePlaySnapshot): LivePlayDiagnosticsSnapshot | null {
  const scope = session.sessionId && session.instanceId ? `${session.sessionId}:${session.instanceId}` : '';
  const [snapshot, setSnapshot] = useState<LivePlayDiagnosticsSnapshot | null>(null);
  useEffect(() => {
    if (!scope || !window.arc?.livePlay) return;
    let alive = true;
    let eventReceived = false;
    const unsubscribe = window.arc.livePlay.onDiagnostics((next) => {
      if (alive && sameSession(next, session)) {
        eventReceived = true;
        setSnapshot(next);
      }
    });
    void window.arc.livePlay
      .diagnosticsSnapshot()
      .then((next) => {
        if (alive && !eventReceived && next && sameSession(next, session)) setSnapshot(next);
      })
      .catch(() => {});
    return () => {
      alive = false;
      unsubscribe();
    };
    // Session and instance IDs are the complete immutable scope, not its mutable connection state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope]);
  return snapshot && sameSession(snapshot, session) ? snapshot : null;
}

const formatTime = (timestamp: number | null): string =>
  timestamp === null ? 'Not received' : new Date(timestamp).toLocaleTimeString();

const metric = (value: number | null, suffix = ''): string =>
  value === null ? '—' : `${value.toLocaleString()}${suffix}`;

export function LivePlayDiagnosticsPanel({ snapshot }: { snapshot: LivePlayDiagnosticsSnapshot | null }) {
  const [warningsOnly, setWarningsOnly] = useState(false);
  const entries = snapshot?.events ?? [];
  const visible = warningsOnly
    ? entries.filter((entry) => entry.level === 'warning' || entry.level === 'error')
    : entries;
  return (
    <section className="live-play-diagnostics" aria-label="Live Play session diagnostics">
      <header className="live-play-diagnostics-header">
        <strong>Session Diagnostics</strong>
        <span>External Development game · read-only</span>
      </header>
      {!snapshot ? (
        <p className="live-runtime-empty">Waiting for a connected session diagnostics snapshot…</p>
      ) : (
        <>
          <dl className="live-play-diagnostics-grid">
            <div>
              <dt>Connection</dt>
              <dd>{snapshot.connection}</dd>
            </div>
            <div>
              <dt>Game state</dt>
              <dd>{snapshot.runtimeState ?? 'Not reported'}</dd>
            </div>
            <div>
              <dt>Tick</dt>
              <dd>{metric(snapshot.runtimeTick)}</dd>
            </div>
            <div>
              <dt>Last heartbeat</dt>
              <dd>{formatTime(snapshot.lastHeartbeatAtMs)}</dd>
            </div>
            <div>
              <dt>Receive sequence</dt>
              <dd>{metric(snapshot.lastReceivedSequence)}</dd>
            </div>
            <div>
              <dt>Content</dt>
              <dd>
                {snapshot.contentStatus && snapshot.contentGeneration !== null
                  ? `${snapshot.contentStatus} · gen ${snapshot.contentGeneration}`
                  : 'Not reported'}
              </dd>
            </div>
            <div>
              <dt>Edit ACK round trip</dt>
              <dd>{metric(snapshot.lastAckLatencyMs, ' ms')}</dd>
            </div>
            <div>
              <dt>Inspection round trip</dt>
              <dd>{metric(snapshot.lastInspectionLatencyMs, ' ms')}</dd>
            </div>
            <div>
              <dt>Applied / rejected</dt>
              <dd>
                {snapshot.appliedCommands} / {snapshot.rejectedCommands}
              </dd>
            </div>
            <div>
              <dt>Truncated/dropped log lines</dt>
              <dd>{snapshot.droppedLogLines}</dd>
            </div>
          </dl>
          <div className="live-play-diagnostics-capabilities">
            <span>Negotiated permissions</span>
            <p>{snapshot.negotiatedCapabilities.join(' · ') || 'None / not connected'}</p>
          </div>
          <div className="live-play-diagnostics-toolbar">
            <strong>Runtime events ({entries.length})</strong>
            <label>
              <input
                type="checkbox"
                checked={warningsOnly}
                onChange={(event) => setWarningsOnly(event.target.checked)}
              />
              Warnings only
            </label>
          </div>
          <div className="live-play-diagnostics-events" role="log" aria-live="off">
            {visible.slice(-80).map((entry) => (
              <div className={`live-play-diagnostic-entry level-${entry.level}`} key={entry.id}>
                <time>{formatTime(entry.timestampMs)}</time>
                <span>{entry.kind}</span>
                <p>{entry.message}</p>
              </div>
            ))}
            {visible.length === 0 && <p className="live-runtime-empty">No matching runtime events.</p>}
          </div>
          <p className="live-play-diagnostics-footnote">
            ACK times measure editor-to-runtime response, not visible frame latency. Disconnected values are historical.
          </p>
        </>
      )}
    </section>
  );
}
