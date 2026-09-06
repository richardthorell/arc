import { useEffect, useMemo, useRef, useState } from 'react';
import { RotateCcw, Settings, X } from 'lucide-react';

import type {
  EditorSettingDescriptor,
  EditorSettingsSnapshot,
  RecoverySnapshot,
} from '../../../common/editorWorkflowTypes';
import type { ArcExtensionSnapshot } from '../../../common/extensionTypes';
import { UiButton, UiIconButton, UiSearchInput, UiSelect, UiTextInput, UiTreeView } from '../ui';
import type { UiTreeNode } from '../ui';
import { defaultExpandedSettingsNodes, editorSettingsNavigation, getEditorSettingsPage } from './settingsNavigation';

import '../tools/tools.css';
import './SettingsDialog.css';

type SettingsDialogProps = {
  onClose: () => void;
  onResetLayout: () => void;
};

const normalize = (value: string) => value.trim().toLocaleLowerCase();

const descriptorSearchTerms = (descriptor: EditorSettingDescriptor) =>
  [descriptor.key, descriptor.label, descriptor.description].join(' ');

const enrichNavigation = (nodes: readonly UiTreeNode[], schema: readonly EditorSettingDescriptor[]): UiTreeNode[] =>
  nodes.map((node) => {
    const page = getEditorSettingsPage(node.id);
    const descriptorKeywords = page?.legacySection
      ? schema.filter((descriptor) => descriptor.section === page.legacySection).map(descriptorSearchTerms)
      : [];
    return {
      ...node,
      keywords: [...(node.keywords ?? []), ...descriptorKeywords],
      children: node.children ? enrichNavigation(node.children, schema) : undefined,
    };
  });

export function SettingsDialog({ onClose, onResetLayout }: SettingsDialogProps) {
  const [snapshot, setSnapshot] = useState<EditorSettingsSnapshot | null>(null);
  const [pageId, setPageId] = useState('general');
  const [scope, setScope] = useState<'user' | 'project'>('user');
  const [query, setQuery] = useState('');
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
  }, []);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [onClose]);

  const page = getEditorSettingsPage(pageId) ?? getEditorSettingsPage('general')!;
  const normalizedQuery = normalize(query);
  const navigation = useMemo(
    () => enrichNavigation(editorSettingsNavigation, snapshot?.schema ?? []),
    [snapshot?.schema],
  );
  const entries = useMemo(() => {
    if (!page.legacySection) return [];
    return (snapshot?.schema ?? []).filter((descriptor) => {
      if (descriptor.section !== page.legacySection) return false;
      if (!normalizedQuery) return true;
      return normalize(descriptorSearchTerms(descriptor)).includes(normalizedQuery);
    });
  }, [normalizedQuery, page.legacySection, snapshot]);

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

  const editor = (descriptor: EditorSettingDescriptor, value: unknown) => {
    const { key } = descriptor;
    const disabled = !descriptor.scopes.includes(scope);
    if (descriptor.type === 'enum')
      return (
        <UiSelect
          ariaLabel={descriptor.label}
          className="settings-value-control"
          disabled={disabled}
          onValueChange={(nextValue) => void update(key, nextValue)}
          options={(descriptor.options ?? []).map((option) => ({ label: option, value: option }))}
          value={String(value)}
        />
      );
    if (typeof value === 'boolean')
      return (
        <input
          aria-label={descriptor.label}
          checked={value}
          className="settings-checkbox"
          disabled={disabled}
          onChange={(event) => void update(key, event.target.checked)}
          type="checkbox"
        />
      );
    if (typeof value === 'number')
      return (
        <UiTextInput
          aria-label={descriptor.label}
          className="settings-value-control"
          disabled={disabled}
          max={descriptor.maximum}
          min={descriptor.minimum}
          onBlur={(event) => void update(key, Number(event.target.value))}
          defaultValue={String(value)}
          key={`${key}-${String(value)}`}
          step={descriptor.step}
          type="number"
        />
      );
    return (
      <UiTextInput
        aria-label={descriptor.label}
        className="settings-value-control"
        disabled={disabled}
        onBlur={(event) => void update(key, event.target.value)}
        defaultValue={String(value)}
        key={`${key}-${String(value)}`}
      />
    );
  };

  const showEmptyPage = entries.length === 0 && page.id !== 'system.recovery' && page.id !== 'tools.extensions';

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
              <strong id="settings-dialog-title">Editor Settings</strong>
              <small>User, machine and project preferences</small>
            </div>
          </div>
          <label className="settings-dialog-scope">
            <span>Scope</span>
            <UiSelect
              ariaLabel="Settings scope"
              onValueChange={(value) => setScope(value as 'user' | 'project')}
              options={[
                { label: 'User settings', value: 'user' },
                { label: 'Project settings', value: 'project' },
              ]}
              value={scope}
            />
          </label>
          <UiIconButton label="Close settings" onClick={onClose} ref={closeButtonRef}>
            <X size={15} />
          </UiIconButton>
        </header>

        <div className="settings-dialog-body">
          <div className="settings-layout">
            <aside className="settings-navigation">
              <UiSearchInput
                aria-label="Search settings"
                autoFocus={false}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search settings"
                value={query}
              />
              <UiTreeView
                ariaLabel="Settings sections"
                defaultExpandedIds={defaultExpandedSettingsNodes}
                nodes={navigation}
                onSelect={(node) => {
                  if (getEditorSettingsPage(node.id)) setPageId(node.id);
                }}
                query={query}
                selectedId={page.id}
              />
            </aside>

            <div className="settings-fields">
              <header className="settings-page-header">
                <h2>{page.label}</h2>
                <p>{page.description}</p>
              </header>

              {entries.map((descriptor) => (
                <div className="settings-field-row" key={descriptor.key}>
                  <span className="settings-field-description">
                    <strong>{descriptor.label}</strong>
                    <small>
                      {descriptor.description}
                      <br />
                      {snapshot?.sources[descriptor.key]}
                      {snapshot?.restartRequired.includes(descriptor.key) ? ' · restart required' : ''}
                      {!descriptor.scopes.includes(scope) ? ` · unavailable in ${scope} settings` : ''}
                    </small>
                  </span>
                  {editor(descriptor, snapshot?.values[descriptor.key])}
                  <UiIconButton
                    label={`Reset ${descriptor.key}`}
                    disabled={!descriptor.scopes.includes(scope)}
                    onClick={() => void update(descriptor.key, undefined)}
                  >
                    <RotateCcw size={13} />
                  </UiIconButton>
                </div>
              ))}

              {page.id === 'general' && (
                <UiButton onClick={onResetLayout} variant="toolbar">
                  Reset workbench layout
                </UiButton>
              )}

              {page.id === 'system.recovery' && (
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

              {page.id === 'tools.extensions' && (
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

              {showEmptyPage && (
                <div className="settings-empty-page">
                  <strong>
                    {normalizedQuery ? 'No matching settings on this page' : 'No settings registered yet'}
                  </strong>
                  <span>
                    {normalizedQuery
                      ? 'Choose another matching category from the tree or clear the search.'
                      : 'This category is ready for settings to be registered in a follow-up stage.'}
                  </span>
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
