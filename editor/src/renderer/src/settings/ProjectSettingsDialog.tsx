import { useCallback, useEffect, useRef, useState } from 'react';
import type { ArcProjectAssetReference, ArcProjectDescriptor } from '../../../common/projectTypes';
import {
  validateProjectSettingChanges,
  visibleProjectSettingsCategories,
  type ProjectSettingsSnapshot,
  type ProjectSettingsField,
  type ProjectSettingsCategory,
} from '../../../common/projectSettingsTypes';
import type { UiTreeNode } from '../ui';
import {
  UiButton,
  UiDialogSettings,
  UiSettingsNavigation,
  UiSettingsCard,
  UiSettingsHeader,
  UiTextInput,
  UiToggleButton,
} from '../ui';
import generalSettingsHeader from './assets/general-settings-header.webp';
import type { AssetPickerItem } from '../inspector/AssetPicker';
import { UiFileImagePicker } from '../ui/UiFileImagePicker';
import { Info, Play } from 'lucide-react';
import { UiTextArea } from '../ui';
import {
  ProjectProfileList,
  ProjectSceneList,
  ProjectSceneSetting,
  ProjectTargetList,
} from './ProjectSettingsCollections';
import './SettingsDialog.css';
import './ProjectSettingsDialog.css';
import { cachedProjectSettings, cacheProjectSettings, refreshProjectSettings } from './projectSettingsCache';

type State = {
  snapshot: ProjectSettingsSnapshot;
  draft: Record<string, unknown>;
  replacement?: ProjectSettingsSnapshot;
};
const equal = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const navigationNodes = (categories: ProjectSettingsCategory[], parentId?: string): UiTreeNode[] =>
  categories
    .filter((entry) => entry.parentId === parentId && !entry.panelOnly)
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
  const [state, setState] = useState<State | undefined>(() => {
    const snapshot = cachedProjectSettings();
    return snapshot ? { snapshot, draft: structuredClone(snapshot.values) } : undefined;
  });
  const [selectedCategory, setCategory] = useState('general-root');
  const categories = state
    ? visibleProjectSettingsCategories(
        state.snapshot.catalog.categories,
        (state.draft.targetPlatforms ?? []) as ArcProjectDescriptor['targetPlatforms'],
      )
    : [];
  const category = categories.some((entry) => entry.id === selectedCategory)
    ? selectedCategory
    : state
      ? 'platforms'
      : 'general-root';
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
      const snapshot = await refreshProjectSettings();
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
                  asset.guid &&
                  /\.(arcscene|png|jpe?g|webp)$/i.test(asset.path) &&
                  (!asset.scope || asset.scope === 'project'),
              )
              .map((asset) => ({
                id: asset.guid,
                guid: asset.guid,
                name: asset.path.split('/').at(-1) ?? asset.path,
                path: asset.path,
                sourcePath: asset.sourcePath,
                kind: asset.path.endsWith('.arcscene') ? 'scene' : 'texture',
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
      Object.entries(state.draft).filter(
        ([key, value]) => key !== '_thumbnailUpload' && !equal(value, state.snapshot.values[key]),
      ),
    );
    const thumbnailUpload = state.draft._thumbnailUpload as string | undefined;
    if (thumbnailUpload) changes.thumbnail = 'Config/ProjectThumbnail.webp';
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
        ...(thumbnailUpload ? { thumbnailUpload } : {}),
      });
      if (!mounted.current) return;
      if (result.succeeded && result.snapshot) {
        cacheProjectSettings(result.snapshot);
        setState({ snapshot: result.snapshot, draft: structuredClone(result.snapshot.values) });
      } else setErrors(result.errors ?? { _form: 'Settings were not applied' });
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
      case 'multiline':
        return (
          <UiTextArea
            aria-label={field.label}
            className="settings-value-control ui-text-input project-description-input"
            rows={3}
            value={String(value ?? '')}
            onChange={(event) => update(event.target.value)}
          />
        );
      case 'image': {
        return (
          <UiFileImagePicker
            key={state.snapshot.projectGuid + state.snapshot.revision}
            preview={(state.draft._thumbnailUpload as string) || (value ? state.snapshot.thumbnailPreview : undefined)}
            hasValue={Boolean(value)}
            browse={() => window.arc.settings.project.browseThumbnail()}
            onChange={(image) =>
              setState((previous) => {
                if (!previous) return previous;
                const draft = { ...previous.draft, thumbnail: image ? 'Config/ProjectThumbnail.webp' : null };
                if (image) Object.assign(draft, { _thumbnailUpload: image });
                else delete (draft as Record<string, unknown>)._thumbnailUpload;
                return { ...previous, draft };
              })
            }
            onError={(message) => setErrors((previous) => ({ ...previous, thumbnail: message }))}
          />
        );
      }
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
            emptyLabel={field.emptyLabel}
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
    state?.snapshot.catalog.fields
      .filter(
        (field) =>
          !field.requiresServerTarget ||
          (Array.isArray(state.draft.targetPlatforms) &&
            (state.draft.targetPlatforms as ArcProjectDescriptor['targetPlatforms']).some(
              (target) => target.enabled && target.id.endsWith('-headless'),
            )),
      )
      .filter((field) =>
        search.trim()
          ? `${field.label} ${field.id} ${field.description ?? ''} ${state.snapshot.catalog.categories.find((entry) => entry.id === field.category)?.label ?? ''}`
              .toLowerCase()
              .includes(search.toLowerCase().trim())
          : category === 'platforms'
            ? field.category === category
            : withinCategory(state.snapshot.catalog.categories, field.category, category),
      ) ?? [];
  return (
    <UiDialogSettings
      onClose={close}
      title="Project Settings"
      sidebar={
        state && (
          <UiSettingsNavigation
            nodes={navigationNodes(categories)}
            defaultExpandedIds={['platforms', 'distribution']}
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
        {state && (
          <UiSettingsHeader
            background={<img alt="" src={generalSettingsHeader} />}
            title={search.trim() ? 'Search results' : categories.find((entry) => entry.id === category)?.label}
            subtitle={
              search.trim()
                ? 'Find project settings across all categories.'
                : categories.find((entry) => entry.id === category)?.description
            }
          />
        )}
        {categories.find((entry) => entry.id === category)?.platform && !search.trim() && (
          <UiSettingsCard
            title="Enabled targets"
            subtitle="Project configuration, not proof of local build or deployment availability."
          >
            <ul>
              {(state?.draft.targetPlatforms as ArcProjectDescriptor['targetPlatforms'])
                .filter((target) => target.enabled && target.id.startsWith(`${category}-`))
                .map((target) => (
                  <li key={target.id}>{target.id}</li>
                ))}
            </ul>
          </UiSettingsCard>
        )}
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
        {state?.snapshot.capabilitiesPending && ['platforms', 'build'].includes(category) && (
          <p role="status">Loading supported target options…</p>
        )}
        {state?.snapshot.capabilitiesError && <p role="alert">{state.snapshot.capabilitiesError}</p>}
        {state && !state.snapshot.writable && <p role="status">This project is read-only.</p>}
        {state && category === 'platforms' && (
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
              <UiSettingsCard
                title={entry.label}
                subtitle={['general', 'startup'].includes(entry.id) ? undefined : entry.description}
                icon={
                  entry.id === 'general' ? <Info size={18} /> : entry.id === 'startup' ? <Play size={18} /> : undefined
                }
                key={entry.id}
              >
                {entries.map((field, index) => (
                  <div
                    key={field.id}
                    className={`settings-field-row${['scenes', 'targets', 'profiles'].includes(field.control) ? ' project-settings-collection-field' : ''}`}
                  >
                    {field.section && field.section !== entries[index - 1]?.section && (
                      <h3 className="project-settings-section-title">{field.section}</h3>
                    )}
                    <span className="settings-field-description">
                      <strong title={field.label}>{field.label}</strong>
                      {field.description && <small title={field.description}>{field.description}</small>}
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
