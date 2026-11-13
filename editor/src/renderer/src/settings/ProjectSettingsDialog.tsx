import { useCallback, useEffect, useRef, useState } from 'react';
import type { ArcProjectAssetReference, ArcProjectDescriptor } from '../../../common/projectTypes';
import {
  validateProjectSettingChanges,
  type ProjectSettingsSnapshot,
  type ProjectSettingsField,
  type ProjectSettingsCategory,
} from '../../../common/projectSettingsTypes';
import type { UiTreeNode } from '../ui';
import { UiButton, UiDialogSettings, UiSettingsNavigation, UiSettingsCard, UiTextInput, UiToggleButton } from '../ui';
import type { AssetPickerItem } from '../inspector/AssetPicker';
import {
  ProjectProfileList,
  ProjectSceneList,
  ProjectSceneSetting,
  ProjectTargetList,
} from './ProjectSettingsCollections';
import './SettingsDialog.css';
import './ProjectSettingsDialog.css';

type State = {
  snapshot: ProjectSettingsSnapshot;
  draft: Record<string, unknown>;
  replacement?: ProjectSettingsSnapshot;
};
const equal = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const navigationNodes = (categories: ProjectSettingsCategory[], parentId?: string): UiTreeNode[] =>
  categories
    .filter((entry) => entry.parentId === parentId)
    .map((entry) => ({
      id: entry.id,
      label: entry.planned ? `${entry.label} (Planned)` : entry.label,
      children: navigationNodes(categories, entry.id),
    }));
const withinCategory = (categories: ProjectSettingsCategory[], candidate: string, selected: string): boolean => {
  const visited = new Set<string>();
  let current: string | undefined = candidate;
  while (current && !visited.has(current)) {
    if (current === selected) return true;
    visited.add(current);
    current = categories.find((entry) => entry.id === current)?.parentId;
  }
  return false;
};

export function ProjectSettingsDialog({ onClose }: { onClose: () => void }) {
  const [state, setState] = useState<State>();
  const [category, setCategory] = useState('general');
  const [search, setSearch] = useState('');
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [confirmClose, setConfirmClose] = useState(false);
  const [assets, setAssets] = useState<AssetPickerItem[]>([]);
  const [tools, setTools] = useState<{ key: string; valid: boolean; message: string }[]>([]);
  const mounted = useRef(false);
  const applying = useRef(false);
  const reading = useRef(false);
  const generation = useRef(0);
  const refresh = useCallback(async () => {
    if (reading.current || applying.current) return;
    reading.current = true;
    const token = generation.current;
    try {
      const api = window.arc?.settings?.project;
      if (!api) throw new Error('Project Settings requires an updated editor host');
      const snapshot = await api.read();
      if (!mounted.current || token !== generation.current) return;
      setState((previous) => {
        if (previous?.snapshot.revision === snapshot.revision) return { ...previous, snapshot };
        if (previous && !equal(previous.draft, previous.snapshot.values)) return { ...previous, replacement: snapshot };
        return { snapshot, draft: structuredClone(snapshot.values) };
      });
      setErrors((previous) => {
        const next = { ...previous };
        delete next._read;
        return next;
      });
    } catch (error) {
      if (mounted.current && token === generation.current)
        setErrors((previous) => ({ ...previous, _read: String(error) }));
    } finally {
      reading.current = false;
    }
  }, []);
  useEffect(() => {
    mounted.current = true;
    const generationRef = generation;
    void refresh();
    const interval = window.setInterval(() => void refresh(), 2000);
    const unsubscribe = window.arc?.settings?.project?.onChanged?.(() => void refresh());
    return () => {
      mounted.current = false;
      ++generationRef.current;
      window.clearInterval(interval);
      unsubscribe?.();
    };
  }, [refresh]);
  useEffect(() => {
    let disposed = false;
    setAssets([]);
    if (state?.snapshot.projectGuid) {
      void window.arc?.host
        ?.query?.('project.assets')
        .then((raw) => {
          if (disposed) return;
          const response = raw as {
            succeeded?: boolean;
            payload?: { assets?: { guid: string; path: string; sourcePath?: string; scope?: string }[] };
          };
          setAssets(
            (response?.payload?.assets ?? [])
              .filter(
                (asset) =>
                  asset.guid && asset.path.endsWith('.arcscene') && (!asset.scope || asset.scope === 'project'),
              )
              .map((asset) => ({
                id: asset.guid,
                guid: asset.guid,
                name: asset.path.split('/').at(-1) ?? asset.path,
                path: asset.path,
                sourcePath: asset.sourcePath,
                kind: 'scene',
                status: 'ready',
              })),
          );
        })
        .catch(() => {});
    }
    return () => {
      disposed = true;
    };
  }, [state?.snapshot.projectGuid, state?.snapshot.revision]);
  useEffect(() => {
    let disposed = false;
    void window.arc?.settings
      ?.snapshot?.()
      .then((snapshot) => {
        if (!disposed)
          setTools(
            Object.entries(snapshot?.pathValidation ?? {})
              .filter(([key]) => ['platform.android.sdkPath', 'platform.windows.visualStudioPath'].includes(key))
              .map(([key, value]) => ({ key, valid: value.valid, message: value.message })),
          );
      })
      .catch(() => {});
    return () => {
      disposed = true;
    };
  }, []);
  const dirty = state && !equal(state.draft, state.snapshot.values);
  const close = () => {
    if (busy) return;
    if (dirty) setConfirmClose(true);
    else onClose();
  };
  const change = (id: string, value: unknown) => {
    setState((previous) => (previous ? { ...previous, draft: { ...previous.draft, [id]: value } } : previous));
    setErrors((previous) => {
      const next = { ...previous };
      delete next[id];
      delete next._form;
      return next;
    });
  };
  const apply = async () => {
    if (!state || applying.current || state.replacement) return;
    const changes = Object.fromEntries(
      Object.entries(state.draft).filter(([key, value]) => !equal(value, state.snapshot.values[key])),
    );
    const invalid = validateProjectSettingChanges(changes);
    setErrors(invalid);
    if (Object.keys(invalid).length) return;
    applying.current = true;
    setBusy(true);
    ++generation.current;
    try {
      const result = await window.arc.settings.project.apply({
        catalogVersion: state.snapshot.catalog.version,
        projectGuid: state.snapshot.projectGuid,
        revision: state.snapshot.revision,
        changes,
      });
      if (!mounted.current) return;
      if (result.succeeded && result.snapshot)
        setState({ snapshot: result.snapshot, draft: structuredClone(result.snapshot.values) });
      else setErrors(result.errors ?? { _form: 'Settings were not applied' });
    } catch (error) {
      if (mounted.current) setErrors({ _form: String(error) });
    } finally {
      applying.current = false;
      if (mounted.current) setBusy(false);
    }
  };
  const fieldControl = (field: ProjectSettingsField) => {
    if (!state) return null;
    const value = state.draft[field.id];
    const update = (next: unknown) => change(field.id, next);
    switch (field.control) {
      case 'readonly':
        return (
          <output className="project-settings-readonly">
            {typeof value === 'object' && value !== null ? JSON.stringify(value, null, 2) : String(value ?? '')}
          </output>
        );
      case 'boolean':
        return (
          <UiToggleButton
            aria-label={field.label}
            className="settings-toggle-control"
            checked={value === true}
            onCheckedChange={update}
          />
        );
      case 'scene':
        return (
          <ProjectSceneSetting
            value={value as ArcProjectAssetReference | null}
            assets={assets}
            label={field.label}
            onChange={update}
          />
        );
      case 'scenes':
        return <ProjectSceneList value={value as ArcProjectAssetReference[]} assets={assets} onChange={update} />;
      case 'targets':
        return (
          <ProjectTargetList
            value={value as ArcProjectDescriptor['targetPlatforms']}
            supported={state.snapshot.capabilities.targets}
            onChange={update}
          />
        );
      case 'profiles':
        return (
          <ProjectProfileList
            value={value as ArcProjectDescriptor['cookProfiles']}
            snapshot={state.snapshot}
            targets={state.draft.targetPlatforms as ArcProjectDescriptor['targetPlatforms']}
            onChange={update}
          />
        );
      default:
        return (
          <UiTextInput
            className="settings-value-control"
            aria-label={field.label}
            aria-invalid={Boolean(errors[field.id])}
            type={field.control === 'number' ? 'number' : 'text'}
            value={String(value ?? '')}
            onChange={(event) => update(field.control === 'number' ? Number(event.target.value) : event.target.value)}
          />
        );
    }
  };
  const fields =
    state?.snapshot.catalog.fields.filter((field) =>
      search.trim()
        ? `${field.label} ${field.id} ${field.description ?? ''} ${state.snapshot.catalog.categories.find((entry) => entry.id === field.category)?.label ?? ''}`
            .toLowerCase()
            .includes(search.toLowerCase().trim())
        : withinCategory(state.snapshot.catalog.categories, field.category, category),
    ) ?? [];
  return (
    <UiDialogSettings
      onClose={close}
      subtitle="Settings shared by this project"
      title="Project Settings"
      sidebar={
        state && (
          <UiSettingsNavigation
            nodes={navigationNodes(state.snapshot.catalog.categories)}
            defaultExpandedIds={['general-root', 'platforms', 'android', 'distribution']}
            selectedId={category}
            query={search}
            onQueryChange={setSearch}
            onSelect={(node) => setCategory(node.id)}
          />
        )
      }
      message={
        <div className="project-settings-footer">
          {confirmClose ? (
            <>
              <span>Discard unapplied project settings?</span>
              <UiButton onClick={onClose}>Discard changes</UiButton>
              <UiButton onClick={() => setConfirmClose(false)}>Keep editing</UiButton>
            </>
          ) : (
            <>
              <UiButton disabled={busy} onClick={close}>
                Cancel
              </UiButton>
              <UiButton
                variant="primary"
                disabled={
                  busy || !dirty || !state?.snapshot.writable || Boolean(state?.replacement) || Boolean(errors._read)
                }
                onClick={() => void apply()}
              >
                {busy ? 'Applying…' : 'Apply'}
              </UiButton>
            </>
          )}
        </div>
      }
    >
      <div className={`settings-fields project-settings-content${!state && !errors._read ? ' is-loading' : ''}`}>
        {Object.entries(errors)
          .filter(([key]) => key.startsWith('_'))
          .map(([key, error]) => (
            <p role="alert" key={key}>
              {error}
            </p>
          ))}
        {state?.replacement && (
          <p role="alert">
            Project settings changed externally. Your draft has not been overwritten.{' '}
            <UiButton
              onClick={() => {
                setState({ snapshot: state.replacement!, draft: structuredClone(state.replacement!.values) });
                setErrors({});
              }}
            >
              Reload and discard draft
            </UiButton>
          </p>
        )}
        {!state && !errors._read && <p role="status">Loading project settings…</p>}
        {state?.snapshot.catalog.categories
          .filter(
            (entry) =>
              entry.description &&
              (search.trim()
                ? `${entry.label} ${entry.description}`.toLowerCase().includes(search.toLowerCase().trim())
                : entry.id === category),
          )
          .map((entry) => (
            <UiSettingsCard key={entry.id} title={entry.planned ? `${entry.label} — Planned` : entry.label}>
              <p role="status">{entry.description}</p>
            </UiSettingsCard>
          ))}
        {state?.snapshot.capabilitiesPending && ['targets', 'build'].includes(category) && (
          <p role="status">Loading supported target options…</p>
        )}
        {state?.snapshot.capabilitiesError && <p role="alert">{state.snapshot.capabilitiesError}</p>}
        {state && !state.snapshot.writable && <p role="status">This project is read-only.</p>}
        {category === 'targets' && (
          <UiSettingsCard
            title="Local tool availability"
            subtitle="Configured in Editor Preferences. These indicators do not imply packaging or launch support."
          >
            {tools.length ? (
              tools.map((tool) => (
                <p key={tool.key}>
                  {tool.key.startsWith('platform.android') ? 'Android' : 'Windows'}:{' '}
                  {tool.valid ? 'Detected' : 'Unavailable'} — {tool.message}
                </p>
              ))
            ) : (
              <p>Local tool information unavailable.</p>
            )}
          </UiSettingsCard>
        )}
        <fieldset disabled={busy || !state?.snapshot.writable}>
          {state?.snapshot.catalog.categories.map((entry) => {
            const entries = fields.filter((field) => field.category === entry.id);
            if (!entries.length) return null;
            return (
              <UiSettingsCard title={entry.label} key={entry.id}>
                {entries.map((field) => (
                  <div
                    key={field.id}
                    className={`settings-field-row${['scenes', 'targets', 'profiles'].includes(field.control) ? ' project-settings-collection-field' : ''}`}
                  >
                    <span className="settings-field-description">
                      <strong title={field.label}>{field.label}</strong>
                      <small title={field.description}>{field.description}</small>
                      {errors[field.id] && <span role="alert">{errors[field.id]}</span>}
                    </span>
                    {['targets', 'profiles'].includes(field.control) ? (
                      <fieldset
                        disabled={Boolean(state.snapshot.capabilitiesPending || state.snapshot.capabilitiesError)}
                      >
                        {fieldControl(field)}
                      </fieldset>
                    ) : (
                      fieldControl(field)
                    )}
                  </div>
                ))}
              </UiSettingsCard>
            );
          })}
        </fieldset>
      </div>
    </UiDialogSettings>
  );
}
