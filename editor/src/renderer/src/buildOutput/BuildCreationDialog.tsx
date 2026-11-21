import { useEffect, useState, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Hammer } from 'lucide-react';

import type { CookWorkflowOptions } from '../../../common/projectWorkflowTypes';
import { UiButton, UiDialog, UiDropdown } from '../ui';
import { platformOptions } from '../layout/MainToolbar';

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
  onClose,
  onStarted,
  onError,
}: {
  action: BuildDialogAction;
  projectKey: string;
  hostPlatform: string;
  preferredPlatform: string;
  currentConfiguration: string;
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
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const busy = useRef(false);
  const alive = useRef(true);

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
  // Use the same display names and brand icons as the main editor toolbar.
  // Preserve the native platform IDs for profile lookup and workflow execution.
  const platformChoices = platforms.map(
    (item) =>
      platformOptions.find((option) => option.value === item) ?? {
        value: item,
        label: item,
      },
  );
  const selectedPlatform = legacyOnly
    ? hostPlatform
    : platforms.includes(platform)
      ? platform
      : (platforms[0] ?? hostPlatform);
  const platformLabel = platformChoices.find((item) => item.value === selectedPlatform)?.label ?? selectedPlatform;
  const availableTargets = Array.from(
    new Set(authored.filter((item) => item.platform === selectedPlatform).map((item) => item.targetId)),
  );
  // Primary Build and Cook actions always operate on an authored project
  // target so they use the complete validate/configure/compile/cook/package
  // workflow. Editor-module commands are intentionally limited to the
  // separately named editor-code maintenance actions below.
  const targets = legacyOnly ? [EDITOR_MODULE] : availableTargets;
  const selectedTarget = targets.includes(target) ? target : (targets[0] ?? '');
  const availableProfiles =
    selectedTarget === EDITOR_MODULE
      ? configurations.map((configuration) => ({ id: configuration, label: configuration }))
      : authored.filter((item) => item.platform === selectedPlatform && item.targetId === selectedTarget);
  const selectedProfile = availableProfiles.some((item) => item.id === profile)
    ? profile
    : (availableProfiles[0]?.id ?? '');
  const valid = Boolean(
    selectedTarget && selectedProfile && !submitting && (selectedTarget === EDITOR_MODULE || options),
  );
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
    setSubmitting(true);
    setError('');
    try {
      // Revalidate the authoritative profile/SDK identity before dispatch.
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
      const outcome = pending.then(
        () => ({ state: 'completed' as const }),
        (reason: unknown) => ({ state: 'failed' as const, reason }),
      );
      let started = false;
      // IPC execution resolves only when the complete workflow finishes. Keep
      // this dialog until the main process confirms that it accepted the run,
      // so planning failures remain actionable here instead of disappearing.
      for (let attempt = 0; attempt < 30 && alive.current; attempt++) {
        const immediate = await Promise.race([
          outcome,
          new Promise<null>((resolve) => setTimeout(() => resolve(null), 50)),
        ]);
        if (immediate?.state === 'failed') throw immediate.reason;
        if (immediate?.state === 'completed') {
          started = true;
          break;
        }
        const snapshot =
          selectedTarget === EDITOR_MODULE
            ? await window.arc.build.snapshot()
            : await window.arc.build.workflowSnapshot();
        started =
          selectedTarget === EDITOR_MODULE
            ? Boolean(
                snapshot && ['building', 'configuring', 'cleaning'].includes('state' in snapshot ? snapshot.state : ''),
              )
            : Boolean(snapshot && ('planning' in snapshot ? snapshot.planning || snapshot.run : false));
        if (started) break;
      }
      if (!alive.current) return;
      if (!started) throw new Error('The build service did not confirm that the build started. Try again.');
      void pending.catch((reason: unknown) => onError(String(reason)));
      onStarted?.();
      onClose();
    } catch (reason) {
      if (alive.current) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      busy.current = false;
      if (alive.current) setSubmitting(false);
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
            <Hammer size={14} /> {submitting ? 'Starting…' : actionLabel}
          </UiButton>
        </>
      }
    >
      <div className="build-create-fields">
        <label>
          <span>Platform</span>
          <UiDropdown
            ariaLabel="Build platform"
            value={selectedPlatform}
            disabled={legacyOnly || submitting}
            options={platformChoices}
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
            disabled={legacyOnly || submitting || !targets.length}
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
            disabled={submitting || !availableProfiles.length}
            options={availableProfiles.map((item) => ({ value: item.id, label: item.label }))}
            onValueChange={setProfile}
          />
        </label>
        {selectedTarget !== EDITOR_MODULE && selectedProfile && (
          <p className="build-create-summary">
            {platformLabel} · {selectedTarget} · {selectedProfile}
            <br />
            {action === 'cook'
              ? 'Cooks saved content only; no compilation, packaging or deployment.'
              : 'Build includes compilation, cooking and packaging. No deployment.'}
          </p>
        )}
        {!selectedTarget && !loading && <p role="alert">No compatible build target is configured for this platform.</p>}
        {blocked && <p role="alert">{blocked}</p>}
        {error && <p role="alert">{error}</p>}
      </div>
    </UiDialog>,
    document.body,
  );
}
