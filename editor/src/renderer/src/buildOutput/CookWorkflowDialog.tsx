import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { CookWorkflowOptions, WorkflowExecutionSnapshot } from '../../../common/projectWorkflowTypes';
import type { EditorDocument } from '../editors/editorTypes';
import { UiButton, UiDialog, UiDropdown } from '../ui';

export function CookWorkflowDialog({
  platform,
  documents,
  saveDocument,
  onClose,
}: {
  platform: string;
  documents: readonly EditorDocument[];
  saveDocument: (document: EditorDocument) => Promise<boolean>;
  onClose: () => void;
}) {
  const [options, setOptions] = useState<CookWorkflowOptions | null>(null);
  const [profileId, setProfileId] = useState('');
  const [snapshot, setSnapshot] = useState<WorkflowExecutionSnapshot | null>(null);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [started, setStarted] = useState(false);
  const inFlight = useRef(false);
  const alive = useRef(true);
  const generation = useRef(0);
  const dirty = documents.filter((document) => document.dirty);
  useEffect(() => {
    alive.current = true;
    const version = ++generation.current;
    const current = () => alive.current && generation.current === version;
    setOptions(null);
    setProfileId('');
    let eventReceived = false;
    const unsubscribe = window.arc.build.onWorkflowState((value) => {
      eventReceived = true;
      setSnapshot(value);
    });
    void window.arc.build
      .workflowSnapshot()
      .then((value) => {
        if (current() && !eventReceived) setSnapshot(value);
      })
      .catch((reason: unknown) => {
        if (current()) setError(String(reason));
      });
    void window.arc.build
      .cookOptions()
      .then((value) => {
        if (!current()) return;
        setOptions(value);
        const profiles = value?.profiles.filter((profile) => profile.platform === platform) ?? [];
        // Multiple authored profiles require an explicit choice.
        setProfileId(profiles.length === 1 ? profiles[0].id : '');
        if (!value) setError('Workflow service is unavailable');
      })
      .catch((reason: unknown) => {
        if (current()) setError(String(reason));
      });
    return () => {
      alive.current = false;
      unsubscribe();
    };
  }, [platform]);
  // Cooking needs a matching cooker, not a connected device or deployment SDK.
  // Keep other authored platform profiles available even if their deployment UI is disabled.
  const profiles = options?.profiles ?? [];
  const selected = profiles.find((profile) => profile.id === profileId);
  const running = Boolean(
    snapshot?.planning || snapshot?.run?.state === 'running' || snapshot?.run?.state === 'cancelling',
  );
  const unavailable =
    options?.unavailableReason || (!profiles.length && options ? 'No enabled cook profiles in this project' : '');
  const execute = async (save: boolean) => {
    if (inFlight.current || running || !selected || !options || unavailable) return;
    inFlight.current = true;
    setSubmitting(true);
    setError('');
    // Capture the profile and project before saving; never use a later UI selection.
    const authority = options.authority;
    const version = generation.current;
    const current = () => alive.current && generation.current === version;
    try {
      if (save) {
        for (const document of dirty) {
          if (!current()) return;
          const latest = await window.arc.build.cookOptions();
          if (!current()) return;
          if (latest?.authority !== authority) throw new Error('Project or SDK changed before saving');
          if (!(await saveDocument(document)))
            throw new Error(`Save cancelled or failed: ${document.title}. Cooking did not start.`);
        }
      }
      if (!current()) return;
      setStarted(true);
      const result = await window.arc.build.executeWorkflow(
        {
          action: 'cook',
          profile: selected.id,
          targetId: selected.targetId,
          destination: 'none',
          sessionId: '',
        },
        authority,
      );
      if (current() && result.error) setError(result.error);
    } catch (reason) {
      if (current()) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      inFlight.current = false;
      if (current()) setSubmitting(false);
    }
  };
  const disabled = submitting || running || !selected || Boolean(unavailable);
  return createPortal(
    <UiDialog
      title="Cook Content Only"
      onClose={onClose}
      width={720}
      footer={
        <>
          {!started && (
            <>
              {dirty.length > 0 && (
                <UiButton disabled={disabled} onClick={() => void execute(true)}>
                  Save All and Cook
                </UiButton>
              )}
              <UiButton disabled={disabled} variant="primary" onClick={() => void execute(false)}>
                {dirty.length ? 'Cook Saved Content' : 'Cook Content'}
              </UiButton>
            </>
          )}
          {running && snapshot?.requestId && (
            <UiButton
              onClick={() => {
                void window.arc.build
                  .cancelWorkflow(snapshot.requestId!)
                  .catch((reason: unknown) => setError(String(reason)));
              }}
            >
              Cancel Cook
            </UiButton>
          )}
          <UiButton onClick={onClose}>{submitting || running || started ? 'Close' : 'Cancel'}</UiButton>
        </>
      }
    >
      <p>
        Cooks saved startup content for {selected?.platform ?? platform}. Does not build code, package, deploy, or
        launch. Closing this window does not cancel a running cook.
      </p>
      <UiDropdown
        ariaLabel="Cook profile"
        value={profileId}
        disabled={submitting || running || started}
        onValueChange={setProfileId}
        options={[
          { value: '', label: 'Select a profile', disabled: true },
          ...profiles.map((profile) => ({ value: profile.id, label: profile.label })),
        ]}
      />
      {!options && !error && <p role="status">Loading profiles…</p>}
      {!started && dirty.length > 0 && (
        <p>
          Unsaved documents: {dirty.map((document) => document.title).join(', ')}. Cook Saved Content leaves these edits
          unchanged.
        </p>
      )}
      {(unavailable || error) && <p role="alert">{error || unavailable}</p>}
      {snapshot && (
        <section aria-label="Cook progress">
          <p role="status">{snapshot.planning ? 'Planning' : (snapshot.run?.state ?? 'Idle')}</p>
          {snapshot.run && <p>{snapshot.run.plan.profile}</p>}
          <ol>
            {snapshot.run?.stages.map((stage) => (
              <li key={stage.stage}>
                {stage.stage}: {stage.state}
              </li>
            ))}
          </ol>
          {snapshot.error && <p role="alert">{snapshot.error}</p>}
          {snapshot.run?.error && <p role="alert">{snapshot.run.error}</p>}
          {snapshot.run?.artifacts.map((artifact) => (
            <p key={artifact.manifestPath}>Verified output: {artifact.manifestPath}</p>
          ))}
          <pre style={{ maxHeight: 240, overflow: 'auto', whiteSpace: 'pre-wrap' }}>{snapshot.output.join('')}</pre>
        </section>
      )}
    </UiDialog>,
    document.body,
  );
}
