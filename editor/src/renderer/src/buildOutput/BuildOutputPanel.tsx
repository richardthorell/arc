import { Ban, ExternalLink, Hammer, Play, RefreshCw, Trash2 } from 'lucide-react';

import type { ArcBuildDiagnostic, ArcBuildRequest, ArcBuildSnapshot } from '../../../common/buildTypes';

import './buildOutput.css';

export function BuildOutputPanel({
  snapshot,
  onExecute,
  onOpenDiagnostic,
}: {
  snapshot: ArcBuildSnapshot | null;
  onExecute: (request: ArcBuildRequest) => void;
  onOpenDiagnostic: (diagnostic: ArcBuildDiagnostic) => void;
}) {
  const busy = snapshot ? ['configuring', 'building', 'cleaning'].includes(snapshot.state) : false;
  return (
    <section className="build-output-panel" aria-label="Build Output">
      <header className="build-output-toolbar">
        <button disabled={busy} onClick={() => onExecute({ action: 'configure' })} type="button">
          <Play size={13} />
          Configure
        </button>
        <button disabled={busy} onClick={() => onExecute({ action: 'build' })} type="button">
          <Hammer size={13} />
          Build
        </button>
        <button disabled={busy} onClick={() => onExecute({ action: 'rebuild' })} type="button">
          <RefreshCw size={13} />
          Rebuild
        </button>
        <button disabled={busy} onClick={() => onExecute({ action: 'clean' })} type="button">
          <Trash2 size={13} />
          Clean
        </button>
        <button disabled={!busy} onClick={() => onExecute({ action: 'cancel' })} type="button">
          <Ban size={13} />
          Cancel
        </button>
        <button
          disabled={busy || !snapshot?.reloadRequired}
          onClick={() => onExecute({ action: 'reload' })}
          type="button"
        >
          <RefreshCw size={13} />
          Reload
        </button>
        <button disabled={busy} onClick={() => onExecute({ action: 'openIde', ide: 'vscode' })} type="button">
          <ExternalLink size={13} />
          Open IDE
        </button>
        <span className={`build-state ${snapshot?.state ?? 'idle'}`}>{snapshot?.state ?? 'idle'}</span>
        {snapshot?.buildRequired && <span className="build-notice">Build required</span>}
        {snapshot?.reloadRequired && <span className="build-notice">Reload required</span>}
        {snapshot?.restartRequired && <span className="build-notice error">Editor host restart required</span>}
      </header>
      <div className="build-output-lines">
        {snapshot?.diagnostics.map((diagnostic) => (
          <button
            className={`build-output-line ${diagnostic.severity}`}
            disabled={!diagnostic.file}
            key={diagnostic.sequence}
            onClick={() => onOpenDiagnostic(diagnostic)}
            title={diagnostic.file ? `${diagnostic.file}:${diagnostic.line ?? 1}` : diagnostic.message}
            type="button"
          >
            <span>{diagnostic.severity}</span>
            <code>
              {diagnostic.file
                ? `${diagnostic.file}:${diagnostic.line ?? 1}:${diagnostic.column ?? 1}`
                : diagnostic.category}
            </code>
            <p>{diagnostic.message}</p>
          </button>
        ))}
        {!snapshot?.diagnostics.length && <div className="build-output-empty">Build output will appear here.</div>}
      </div>
    </section>
  );
}
