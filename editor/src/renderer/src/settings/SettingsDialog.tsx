import { useEffect, useMemo, useRef, useState } from 'react';
import { RotateCcw, Settings, X } from 'lucide-react';

import type { EditorSettingsSnapshot, RecoverySnapshot } from '../../../common/editorWorkflowTypes';
import type { ArcExtensionSnapshot } from '../../../common/extensionTypes';
import { UiButton, UiIconButton } from '../ui';

import '../tools/tools.css';
import './SettingsDialog.css';

const sections = [
  'Editor',
  'Renderer',
  'Input',
  'Cache',
  'Paths & Tools',
  'Extensions',
  'Source Control',
  'Recovery',
] as const;

type SettingsDialogProps = {
  onClose: () => void;
  onResetLayout: () => void;
};

export function SettingsDialog({ onClose, onResetLayout }: SettingsDialogProps) {
  const [snapshot, setSnapshot] = useState<EditorSettingsSnapshot | null>(null);
  const [section, setSection] = useState<(typeof sections)[number]>('Editor');
  const [scope, setScope] = useState<'user' | 'project'>('user');
  const [message, setMessage] = useState('');
  const [recovery, setRecovery] = useState<RecoverySnapshot | null>(null);
  const [extensions, setExtensions] = useState<ArcExtensionSnapshot | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    void window.arc.settings.snapshot().then(setSnapshot);
    void window.arc.recovery.snapshot().then(setRecovery);
    void window.arc.extensions.snapshot().then(setExtensions);
  }, []);

  useEffect(() => {
    closeButtonRef.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [onClose]);

  const entries = useMemo(
    () => snapshot?.schema.filter((descriptor) => descriptor.section === section) ?? [],
    [section, snapshot],
  );

  const update = async (key: string, value: unknown) => {
    if (!snapshot) return;
    try {
      const next = await window.arc.settings.update(scope, { [key]: value }, snapshot.revision);
      if (next) setSnapshot(next);
      setMessage(`${key} updated in ${scope} settings`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  };

  const editor = (key: string, value: unknown) => {
    const descriptor = snapshot?.schema.find((entry) => entry.key === key);
    const disabled = !descriptor?.scopes.includes(scope);
    if (descriptor?.type === 'enum')
      return (
        <select disabled={disabled} onChange={(event) => void update(key, event.target.value)} value={String(value)}>
          {descriptor.options?.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      );
    if (typeof value === 'boolean')
      return (
        <input
          checked={value}
          disabled={disabled}
          onChange={(event) => void update(key, event.target.checked)}
          type="checkbox"
        />
      );
    if (typeof value === 'number')
      return (
        <input
          disabled={disabled}
          max={descriptor?.maximum}
          min={descriptor?.minimum}
          onBlur={(event) => void update(key, Number(event.target.value))}
          defaultValue={String(value)}
          key={`${key}-${String(value)}`}
          step={descriptor?.step}
          type="number"
        />
      );
    return (
      <input
        disabled={disabled}
        onBlur={(event) => void update(key, event.target.value)}
        defaultValue={String(value)}
        key={`${key}-${String(value)}`}
      />
    );
  };

  return (
    <div
      className="settings-dialog-backdrop"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section aria-labelledby="settings-dialog-title" aria-modal="true" className="settings-dialog" role="dialog">
        <header className="settings-dialog-titlebar">
          <div className="settings-dialog-title">
            <Settings aria-hidden="true" size={18} />
            <div>
              <strong id="settings-dialog-title">Settings</strong>
              <small>Editor and project preferences</small>
            </div>
          </div>
          <label className="settings-dialog-scope">
            <span>Scope</span>
            <select
              aria-label="Settings scope"
              value={scope}
              onChange={(event) => setScope(event.target.value as 'user' | 'project')}
            >
              <option value="user">User settings</option>
              <option value="project">Project settings</option>
            </select>
          </label>
          <UiIconButton label="Close settings" onClick={onClose} ref={closeButtonRef}>
            <X size={15} />
          </UiIconButton>
        </header>

        <div className="settings-dialog-body">
          <div className="settings-layout">
            <nav aria-label="Settings sections">
              {sections.map((name) => (
                <button
                  className={name === section ? 'active' : ''}
                  key={name}
                  onClick={() => setSection(name)}
                  type="button"
                >
                  {name}
                </button>
              ))}
            </nav>
            <div className="settings-fields">
              <h2>{section}</h2>
              {entries.map((descriptor) => (
                <label key={descriptor.key}>
                  <span>
                    <strong>{descriptor.label}</strong>
                    <small>
                      {descriptor.description}
                      <br />
                      {snapshot?.sources[descriptor.key]}
                      {snapshot?.restartRequired.includes(descriptor.key) ? ' · restart required' : ''}
                      {!descriptor.scopes.includes(scope) ? ` · unavailable in ${scope} settings` : ''}
                    </small>
                  </span>
                  {editor(descriptor.key, snapshot?.values[descriptor.key])}
                  <button
                    aria-label={`Reset ${descriptor.key}`}
                    disabled={!descriptor.scopes.includes(scope)}
                    onClick={() => void update(descriptor.key, undefined)}
                    type="button"
                  >
                    <RotateCcw size={13} />
                  </button>
                </label>
              ))}
              {section === 'Editor' && (
                <UiButton onClick={onResetLayout} variant="toolbar">
                  Reset workbench layout
                </UiButton>
              )}
              {section === 'Recovery' && (
                <div className="recovery-browser">
                  <p>
                    {recovery?.uncleanShutdown
                      ? 'ARC detected an unclean editor shutdown. Recovery generations are available below.'
                      : 'Recovery snapshots are stored outside the project and never overwrite source files.'}
                  </p>
                  {recovery?.generations.map((generation) => (
                    <article key={generation.id}>
                      <span>
                        <strong>{generation.documentName}</strong>
                        <small>
                          {new Date(generation.createdAt).toLocaleString()} · {(generation.size / 1024).toFixed(1)} KiB
                        </small>
                      </span>
                      <UiButton
                        onClick={() =>
                          void window.arc.recovery
                            .restore(generation.id)
                            .then(() => setMessage('Recovery opened as dirty'))
                        }
                        variant="toolbar"
                      >
                        Open
                      </UiButton>
                      <UiButton
                        onClick={() =>
                          void window.arc.recovery.discard(generation.id).then(async () => {
                            setRecovery(await window.arc.recovery.snapshot());
                          })
                        }
                        variant="toolbar"
                      >
                        Discard
                      </UiButton>
                    </article>
                  ))}
                  {!recovery?.generations.length && <div className="tool-empty">No recovery generations.</div>}
                </div>
              )}
              {section === 'Extensions' && (
                <div className="recovery-browser">
                  {extensions?.extensions.map((extension) => (
                    <article key={extension.manifest.id}>
                      <span>
                        <strong>
                          {extension.manifest.name} {extension.manifest.version}
                        </strong>
                        <small>
                          {extension.enabled ? 'Enabled' : 'Disabled'} ·{' '}
                          {extension.manifest.capabilities.join(', ') || 'No capabilities'}
                        </small>
                        {extension.diagnostics.map((diagnostic) => (
                          <small className="tool-error" key={diagnostic}>
                            {diagnostic}
                          </small>
                        ))}
                      </span>
                    </article>
                  ))}
                  {!extensions?.extensions.length && (
                    <div className="tool-empty">No extensions are declared by this project.</div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {message && <div className="settings-dialog-message tool-message">{message}</div>}
      </section>
    </div>
  );
}
