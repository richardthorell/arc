import { useEffect, useMemo, useState } from 'react';
import { Palette, RotateCcw } from 'lucide-react';

import type {
  EditorSettingDescriptor,
  EditorSettingsSnapshot,
  RecoverySnapshot,
} from '../../../common/editorWorkflowTypes';
import type { ArcExtensionSnapshot } from '../../../common/extensionTypes';
import {
  UiButton,
  UiDialogSettings,
  UiIconButton,
  UiSelect,
  UiSettingsCard,
  UiSettingsHeader,
  UiSettingsNavigation,
  UiTextInput,
  UiToggleButton,
} from '../ui';
import type { UiTreeNode } from '../ui';
import { defaultExpandedSettingsNodes, editorSettingsNavigation, getEditorSettingsPage } from './settingsNavigation';

import '../tools/tools.css';
import './SettingsDialog.css';

type EditorPreferencesDialogProps = {
  onClose: () => void;
  onResetLayout: () => void;
};

const normalize = (value: string) => value.trim().toLocaleLowerCase();

const descriptorSearchTerms = (descriptor: EditorSettingDescriptor) =>
  [descriptor.key, descriptor.label, descriptor.description].join(' ');

const enumOptionLabel = (descriptor: EditorSettingDescriptor, option: string) => {
  if (descriptor.key === 'editor.theme' && option === 'arcDark') return 'Dark (Default)';
  return option;
};

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

export function EditorPreferencesDialog({ onClose, onResetLayout }: EditorPreferencesDialogProps) {
  const [snapshot, setSnapshot] = useState<EditorSettingsSnapshot | null>(null);
  const [pageId, setPageId] = useState('general');
  const [query, setQuery] = useState('');
  const [message, setMessage] = useState('');
  const [recovery, setRecovery] = useState<RecoverySnapshot | null>(null);
  const [extensions, setExtensions] = useState<ArcExtensionSnapshot | null>(null);

  useEffect(() => {
    void window.arc.settings.snapshot().then(setSnapshot);
    void window.arc.recovery.snapshot().then(setRecovery);
    void window.arc.extensions.snapshot().then(setExtensions);
  }, []);

  const page = getEditorSettingsPage(pageId) ?? getEditorSettingsPage('general')!;
  const normalizedQuery = normalize(query);
  const userSchema = useMemo(
    () => (snapshot?.schema ?? []).filter((descriptor) => descriptor.scopes.includes('user')),
    [snapshot?.schema],
  );
  const navigation = useMemo(() => enrichNavigation(editorSettingsNavigation, userSchema), [userSchema]);
  const entries = useMemo(() => {
    if (!page.legacySection) return [];
    return userSchema.filter((descriptor) => {
      if (descriptor.section !== page.legacySection) return false;
      if (!normalizedQuery) return true;
      return normalize(descriptorSearchTerms(descriptor)).includes(normalizedQuery);
    });
  }, [normalizedQuery, page.legacySection, userSchema]);

  const update = async (key: string, value: unknown) => {
    if (!snapshot) return;
    try {
      const next = await window.arc.settings.update('user', { [key]: value }, snapshot.revision);
      if (next) {
        setSnapshot(next);
        window.dispatchEvent(new CustomEvent('arc-editor-settings-changed', { detail: next }));
      }
      setMessage(`${key} updated in user settings`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  };

  const editor = (descriptor: EditorSettingDescriptor, value: unknown) => {
    const { key } = descriptor;
    if (descriptor.format === 'color' && typeof value === 'string')
      return (
        <input
          aria-label={descriptor.label}
          className="settings-color-control"
          onChange={(event) => void update(key, event.target.value.toUpperCase())}
          type="color"
          value={value}
        />
      );
    if (descriptor.type === 'enum')
      return (
        <UiSelect
          ariaLabel={descriptor.label}
          className="settings-value-control"
          onValueChange={(nextValue) => void update(key, nextValue)}
          options={(descriptor.options ?? []).map((option) => ({
            label: enumOptionLabel(descriptor, option),
            value: option,
          }))}
          value={String(value)}
        />
      );
    if (typeof value === 'boolean')
      return (
        <UiToggleButton
          aria-label={descriptor.label}
          checked={value}
          className="settings-toggle-control"
          onCheckedChange={(checked) => void update(key, checked)}
        />
      );
    if (typeof value === 'number')
      return (
        <UiTextInput
          aria-label={descriptor.label}
          className="settings-value-control"
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
        onBlur={(event) => void update(key, event.target.value)}
        defaultValue={String(value)}
        key={`${key}-${String(value)}`}
      />
    );
  };

  const showEmptyPage = entries.length === 0 && page.id !== 'system.recovery' && page.id !== 'tools.extensions';

  const sidebar = (
    <UiSettingsNavigation
      defaultExpandedIds={defaultExpandedSettingsNodes}
      nodes={navigation}
      onQueryChange={setQuery}
      onSelect={(node) => {
        if (getEditorSettingsPage(node.id)) setPageId(node.id);
      }}
      query={query}
      searchAriaLabel="Search preferences"
      searchPlaceholder="Search preferences"
      selectedId={page.id}
      treeAriaLabel="Preference sections"
    />
  );

  return (
    <UiDialogSettings
      message={message ? <div className="tool-message">{message}</div> : undefined}
      onClose={onClose}
      sidebar={sidebar}
      subtitle="Personal editor and machine preferences"
      title="Editor Preferences"
    >
      <div className="settings-fields">
        <UiSettingsHeader subtitle={page.description} title={page.label} />

        {entries.length > 0 && (
          <UiSettingsCard
            icon={page.id === 'general' ? <Palette aria-hidden="true" size={16} /> : undefined}
            title={page.id === 'general' ? 'Appearance' : (page.legacySection ?? page.label)}
          >
            {entries.map((descriptor) => (
              <div className="settings-field-row" key={descriptor.key}>
                <span className="settings-field-description">
                  <strong>{descriptor.label}</strong>
                  <small>
                    {descriptor.description}
                    {snapshot?.restartRequired.includes(descriptor.key) ? ' · restart required' : ''}
                  </small>
                </span>
                {editor(descriptor, snapshot?.values[descriptor.key])}
                <UiIconButton label={`Reset ${descriptor.key}`} onClick={() => void update(descriptor.key, undefined)}>
                  <RotateCcw size={13} />
                </UiIconButton>
              </div>
            ))}
          </UiSettingsCard>
        )}

        {page.id === 'general' && (
          <UiSettingsCard subtitle="Restore the default editor panel arrangement." title="Workbench">
            <div className="settings-card-actions">
              <UiButton onClick={onResetLayout} variant="toolbar">
                Reset workbench layout
              </UiButton>
            </div>
          </UiSettingsCard>
        )}

        {page.id === 'system.recovery' && (
          <UiSettingsCard
            subtitle={
              recovery?.uncleanShutdown
                ? 'ARC detected an unclean editor shutdown. Recovery generations are available below.'
                : 'Recovery snapshots are stored outside the project and never overwrite source files.'
            }
            title="Recovery generations"
          >
            <div className="recovery-browser settings-card-list">
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
                      void window.arc.recovery.restore(generation.id).then(() => setMessage('Recovery opened as dirty'))
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
          </UiSettingsCard>
        )}

        {page.id === 'tools.extensions' && (
          <UiSettingsCard subtitle="Extensions declared by the current project." title="Extensions">
            <div className="recovery-browser settings-card-list">
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
          </UiSettingsCard>
        )}

        {showEmptyPage && (
          <div className="settings-empty-page">
            <strong>
              {normalizedQuery ? 'No matching preferences on this page' : 'No preferences registered yet'}
            </strong>
            <span>
              {normalizedQuery
                ? 'Choose another matching category from the tree or clear the search.'
                : 'This category is ready for preferences to be registered in a follow-up stage.'}
            </span>
          </div>
        )}
      </div>
    </UiDialogSettings>
  );
}
