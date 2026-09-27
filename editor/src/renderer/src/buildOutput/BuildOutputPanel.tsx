import { Ban, ExternalLink, Hammer, Play, RefreshCw, Trash2 } from 'lucide-react';
import { useEffect, useState } from 'react';

import type { ArcBuildDiagnostic, ArcBuildRequest, ArcBuildSnapshot } from '../../../common/buildTypes';
import { UiSelect, type UiSelectOption } from '../ui/UiSelect';

import './buildOutput.css';

const DEFAULT_BUILD_CONFIGURATIONS: ReadonlyArray<UiSelectOption> = [
  { value: 'Debug', label: 'Debug' },
  { value: 'RelWithDebInfo', label: 'RelWithDebInfo' },
  { value: 'Release', label: 'Release' },
];

export function BuildOutputPanel({
  snapshot,
  onExecute,
  onOpenDiagnostic,
  configurations = DEFAULT_BUILD_CONFIGURATIONS,
}: {
  snapshot: ArcBuildSnapshot | null;
  onExecute: (request: ArcBuildRequest) => void;
  onOpenDiagnostic: (diagnostic: ArcBuildDiagnostic) => void;
  configurations?: ReadonlyArray<UiSelectOption>;
}) {
  const busy = snapshot ? ['configuring', 'building', 'cleaning'].includes(snapshot.state) : false;
  const fallbackConfiguration = configurations[0]?.value ?? 'Debug';
  const [configuration, setConfiguration] = useState(snapshot?.configuration ?? fallbackConfiguration);

  useEffect(() => {
    if (snapshot?.configuration) setConfiguration(snapshot.configuration);
  }, [snapshot?.configuration]);

  const executeForConfiguration = (action: 'configure' | 'build' | 'rebuild' | 'clean') =>
    onExecute({ action, configuration });

  return (
    <section className="build-output-panel" aria-label="Build Output">
      <header className="build-output-toolbar">
        <UiSelect
          ariaLabel="Build configuration"
          disabled={busy}
          onValueChange={setConfiguration}
          options={configurations}
          value={configuration}
        />
        <button disabled={busy} onClick={() => executeForConfiguration('configure')} type="button">
          <Play size={13} />
          Configure
        </button>
        <button disabled={busy} onClick={() => executeForConfiguration('build')} type="button">
          <Hammer size={13} />
          Build
        </button>
        <button disabled={busy} onClick={() => executeForConfiguration('rebuild')} type="button">
          <RefreshCw size={13} />
          Rebuild
        </button>
        <button disabled={busy} onClick={() => executeForConfiguration('clean')} type="button">
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
