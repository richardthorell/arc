import { useEffect, useState, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Hammer } from 'lucide-react';

import type { CookWorkflowOptions } from '../../../common/projectWorkflowTypes';
import type { EditorDocument } from '../editors/editorTypes';
import { UiButton, UiDialog, UiDropdown } from '../ui';

import type { BuildDialogAction } from './BuildToolbar';

import './BuildCreationDialog.css';

const EDITOR_MODULE = 'editor-module';
const configurations = ['Debug', 'RelWithDebInfo', 'Release'] as const;
type BuildPreference = { platform: string; target: string; profile: string };

const preferenceKey = (projectKey: string) => `arc.build.dialog.v1:${projectKey}`;
const loadPreference = (projectKey: string): BuildPreference | null => {
  try {
    const raw = localStorage.getItem(preferenceKey(projectKey));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<BuildPreference>;
    if (typeof parsed.platform !== 'string' || typeof parsed.target !== 'string' || typeof parsed.profile !== 'string')
      return null;
    return { platform: parsed.platform, target: parsed.target, profile: parsed.profile };
  } catch {
    return null;
  }
};

/**
 * Platform/profile/target are build inputs only. Deployment devices are chosen
 * later by Run, and are never required for package or cook actions.
 */
export function BuildCreationDialog({
  action,
  projectKey,
  hostPlatform,
  preferredPlatform,
  currentConfiguration,
  documents,
  saveDocument,
  onClose,
  onStarted,
  onError,
}: {
  action: BuildDialogAction;
  projectKey: string;
  hostPlatform: string;
  preferredPlatform: string;
  currentConfiguration: string;
  documents: readonly EditorDocument[];
  saveDocument: (document: EditorDocument) => Promise<boolean>;
  onClose: () => void;
  onStarted?: () => void;
  onError: (message: string) => void;
}) {
  const saved = useRef(loadPreference(projectKey));
  const initialPlatform =
    action === 'build' || action === 'cook' ? (saved.current?.platform ?? preferredPlatform) : hostPlatform;

  const [options, setOptions] = useState<CookWorkflowOptions | null>(null);
  const [platform, setPlatform] = useState(initialPlatform);
  const [target, setTarget] = useState(action === 'cook' ? '' : (saved.current?.target ?? ''));
  const [profile, setProfile] = useState(saved.current?.profile ?? currentConfiguration);
  const [saving, setSaving] = useState(false);
  const [saveChanges, setSaveChanges] = useState(true);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const busy = useRef(false);
  const alive = useRef(true);

  const dirty = documents.filter((document) => document.dirty);
  const legacyOnly = action === 'rebuild' || action === 'configure' || action === 'clean';

  useEffect(() => {
    alive.current = true;
    let mounted = true;
    void window.arc.build
      .cookOptions()
      .then((result) => {
        if (mounted) setOptions(result);
      })
      .catch((reason: unknown) => {
        if (mounted) setError(String(reason));
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
      alive.current = false;
    };
  }, []);

  const authored = options?.profiles ?? [];
  const platforms = Array.from(new Set([...authored.map((item) => item.platform), hostPlatform]));
  const selectedPlatform = legacyOnly
    ? hostPlatform
    : platforms.includes(platform)
      ? platform
      : (platforms[0] ?? hostPlatform);
  const availableTargets = Array.from(
    new Set(authored.filter((item) => item.platform === selectedPlatform).map((item) => item.targetId)),
  );
  const targets = [
    ...(action !== 'cook' && selectedPlatform === hostPlatform ? [EDITOR_MODULE] : []),
    ...(legacyOnly ? [] : availableTargets),
  ];
  const selectedTarget = targets.includes(target) ? target : (targets[0] ?? '');
  const availableProfiles =
    selectedTarget === EDITOR_MODULE
      ? configurations.map((configuration) => ({ id: configuration, label: configuration }))
      : authored.filter((item) => item.platform === selectedPlatform && item.targetId === selectedTarget);
  const selectedProfile = availableProfiles.some((item) => item.id === profile)
    ? profile
    : (availableProfiles[0]?.id ?? '');
  const valid = Boolean(selectedTarget && selectedProfile && !saving && (selectedTarget === EDITOR_MODULE || options));
  const authoredSelection = authored.find(
    (item) => item.id === selectedProfile && item.targetId === selectedTarget && item.platform === selectedPlatform,
  );
  const blocked =
    selectedTarget === EDITOR_MODULE
      ? undefined
      : (options?.unavailableReason ??
        (action === 'build' && !authoredSelection?.editorBuildAvailable
          ? (authoredSelection?.editorBuildReason ??
            'This platform/profile cannot be packaged by the current Editor build adapter.')
          : undefined));
  const actionLabel = {
    build: 'Build',
    rebuild: 'Rebuild',
    cook: 'Cook Content',
    configure: 'Configure',
    clean: 'Clean',
  }[action];

  const submit = async () => {
    if (!valid || blocked || busy.current) return;
    busy.current = true;
    setSaving(true);
    setError('');
    try {
      if (dirty.length && saveChanges) {
        for (const document of dirty) {
          if (!(await saveDocument(document))) {
            throw new Error(`Could not save ${document.title}. Build did not start.`);
          }
          if (!alive.current) return;
        }
      }
      // Revalidate the authoritative profile/SDK identity after saving.
      if (selectedTarget !== EDITOR_MODULE) {
        const latest = await window.arc.build.cookOptions();
        if (!alive.current) return;
        if (!options || !latest || latest.authority !== options.authority)
          throw new Error('Project or SDK changed. Reopen Build to refresh profiles.');
        if (
          !latest.profiles.some(
            (item) =>
              item.id === selectedProfile && item.targetId === selectedTarget && item.platform === selectedPlatform,
          )
        )
          throw new Error('The selected build profile is no longer available.');
      }

      if (!alive.current) return;
      const preference = { platform: selectedPlatform, target: selectedTarget, profile: selectedProfile };
      try {
        localStorage.setItem(preferenceKey(projectKey), JSON.stringify(preference));
      } catch {
        // Preferences are optional; never block a build.
      }

      // IPC dispatches into the main-process service. Do not await the entire
      // build in a dialog: closing it must not terminate or pause execution.
      const executeCodeOperation = () => {
        if (action === 'cook') throw new Error('Cooking requires an authored content target');
        return window.arc.build.execute({ action, configuration: selectedProfile });
      };
      const pending =
        selectedTarget === EDITOR_MODULE
          ? executeCodeOperation()
          : window.arc.build.executeWorkflow(
              {
                action: action === 'cook' ? 'cook' : 'build',
                profile: selectedProfile,
                targetId: selectedTarget,
                destination: 'none',
                sessionId: '',
              },
              options?.authority,
            );
      void pending.catch((reason: unknown) => onError(String(reason)));
      onStarted?.();
      onClose();
    } catch (reason) {
      if (alive.current) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      busy.current = false;
      if (alive.current) setSaving(false);
    }
  };

  const targetOptions = targets.map((item) => ({
    value: item,
    label: item === EDITOR_MODULE ? 'Editor module (native code)' : item,
  }));

  return createPortal(
    <UiDialog
      title={`${actionLabel} project`}
      subtitle="Build configuration"
      icon={<Hammer size={17} />}
      onClose={onClose}
      width={510}
      footer={
        <>
          <UiButton onClick={onClose}>Cancel</UiButton>
          <UiButton disabled={!valid || Boolean(blocked)} variant="primary" onClick={() => void submit()}>
            <Hammer size={14} /> {saving ? 'Saving…' : actionLabel}
          </UiButton>
        </>
      }
    >
      <div className="build-create-fields">
        <p>Choose what to {action === 'clean' ? 'clean' : 'build'}. Device selection belongs to Run, not Build.</p>
        <label>
          <span>Platform</span>
          <UiDropdown
            ariaLabel="Build platform"
            value={selectedPlatform}
            disabled={legacyOnly || saving}
            options={platforms.map((item) => ({ value: item, label: item }))}
            onValueChange={(value) => {
              setPlatform(value);
              setTarget('');
              setProfile('');
            }}
          />
        </label>
        <label>
          <span>Build target</span>
          <UiDropdown
            ariaLabel="Build target"
            value={selectedTarget}
            disabled={legacyOnly || saving || !targets.length}
            options={targetOptions}
            onValueChange={(value) => {
              setTarget(value);
              setProfile('');
            }}
          />
        </label>
        <label>
          <span>{selectedTarget === EDITOR_MODULE ? 'Configuration' : 'Build profile'}</span>
          <UiDropdown
            ariaLabel="Build profile"
            value={selectedProfile}
            disabled={saving || !availableProfiles.length}
            options={availableProfiles.map((item) => ({ value: item.id, label: item.label }))}
            onValueChange={setProfile}
          />
        </label>
        {selectedTarget !== EDITOR_MODULE && selectedProfile && (
          <p className="build-create-summary">
            {selectedPlatform} · {selectedTarget} · {selectedProfile}
            <br />
            {action === 'cook'
              ? 'Cooks saved content only; no compilation, packaging or deployment.'
              : 'Build includes compilation, cooking and packaging. No deployment.'}
          </p>
        )}
        {dirty.length > 0 && (
          <label className="build-create-save">
            <input
              type="checkbox"
              checked={saveChanges}
              disabled={saving}
              onChange={(event) => setSaveChanges(event.target.checked)}
            />
            Save {dirty.length} modified document{dirty.length === 1 ? '' : 's'} before {actionLabel.toLowerCase()}
          </label>
        )}
        {dirty.length > 0 && !saveChanges && (
          <p className="build-create-hint">Unsaved changes will not be included in cooked content.</p>
        )}
        {loading && <p role="status">Loading available project profiles…</p>}
        {!selectedTarget && !loading && <p role="alert">No compatible build target is configured for this platform.</p>}
        {blocked && <p role="alert">{blocked}</p>}
        {error && <p role="alert">{error}</p>}
      </div>
    </UiDialog>,
    document.body,
  );
}
