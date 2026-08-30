import { useEffect, useRef } from 'react';

import type { ArcBuildSnapshot } from '../../../common/buildTypes';
import type { StartupState } from '../app/workbenchTypes';
import { beginEditorJob, type EditorJobToken } from './editorJobProgress';

const terminalBuildStep = (state: string) =>
  state === 'succeeded' || state === 'failed' || state === 'cancelled' || state === 'skipped';

const buildFallbackLabel = (snapshot: ArcBuildSnapshot) => {
  if (snapshot.action === 'configure') return 'Configuring project';
  if (snapshot.action === 'test') return 'Running tests';
  if (snapshot.action === 'run') return 'Launching project';
  return 'Building project';
};

const buildActivity = (snapshot: ArcBuildSnapshot) => {
  const runningStep = snapshot.steps.find((step) => step.state === 'running');
  const pendingStep = snapshot.steps.find((step) => step.state === 'pending');
  const label = runningStep?.label || pendingStep?.label || buildFallbackLabel(snapshot);
  const total = snapshot.steps.length;
  const completed = snapshot.steps.filter((step) => terminalBuildStep(step.state)).length;
  return { label, total, completed };
};

const hostActivityLabel = (type: string, message: string) => {
  const normalized = `${type} ${message}`.toLocaleLowerCase();
  if (normalized.includes('cook')) return message || 'Cooking assets';
  if (normalized.includes('shader') && normalized.includes('compil')) return message || 'Compiling shaders';
  if (normalized.includes('import')) return message || 'Importing assets';
  if (type === 'project.opened') return 'Loading assets';
  if (type === 'project.moduleReloaded') return 'Loading project modules';
  if (type === 'asset.changed') return message || 'Processing assets';
  return '';
};

export function useEditorActivityProgress(
  startupState: StartupState | null,
  activeScene?: string,
  enabled = true,
) {
  const startupJob = useRef<EditorJobToken | null>(null);
  const buildJob = useRef<EditorJobToken | null>(null);
  const hostActivityJob = useRef<EditorJobToken | null>(null);
  const hostActivityTimer = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled) {
      startupJob.current?.finish();
      startupJob.current = null;
      return;
    }
    if (!startupState) {
      if (!startupJob.current)
        startupJob.current = beginEditorJob('Starting editor', {
          priority: 'foreground',
          indeterminate: true,
        });
      else startupJob.current.update({ label: 'Starting editor', indeterminate: true });
      return;
    }

    if (startupState.engineHostConnected && !activeScene) {
      if (!startupJob.current)
        startupJob.current = beginEditorJob('Opening scene', {
          priority: 'foreground',
          indeterminate: true,
        });
      else startupJob.current.update({ label: 'Opening scene', indeterminate: true });
      return;
    }

    startupJob.current?.finish();
    startupJob.current = null;
  }, [activeScene, enabled, startupState]);

  useEffect(() => {
    if (!enabled || !window.arc?.build) return;

    const accept = (snapshot: ArcBuildSnapshot | null) => {
      if (!snapshot || (snapshot.status !== 'queued' && snapshot.status !== 'running')) {
        buildJob.current?.finish();
        buildJob.current = null;
        return;
      }

      const activity = buildActivity(snapshot);
      const update =
        activity.total > 0
          ? { label: activity.label, completed: activity.completed, total: activity.total, indeterminate: false }
          : { label: activity.label, completed: 0, total: 0, indeterminate: true };

      if (!buildJob.current) buildJob.current = beginEditorJob(activity.label, { priority: 'foreground', ...update });
      else buildJob.current.update(update);
    };

    void window.arc.build.snapshot().then(accept);
    return window.arc.build.onState(accept);
  }, [enabled]);

  useEffect(() => {
    if (!enabled || !window.arc?.host?.onEvent) return;

    const finishHostActivity = () => {
      hostActivityJob.current?.finish();
      hostActivityJob.current = null;
      if (hostActivityTimer.current !== null) window.clearTimeout(hostActivityTimer.current);
      hostActivityTimer.current = null;
    };

    const scheduleFinish = () => {
      if (hostActivityTimer.current !== null) window.clearTimeout(hostActivityTimer.current);
      hostActivityTimer.current = window.setTimeout(finishHostActivity, 650);
    };

    const unsubscribe = window.arc.host.onEvent((event) => {
      const label = hostActivityLabel(event.type, event.message);
      if (!label) return;

      if (!hostActivityJob.current)
        hostActivityJob.current = beginEditorJob(label, {
          priority: 'normal',
          indeterminate: true,
        });
      else hostActivityJob.current.update({ label, indeterminate: true });
      scheduleFinish();
    });

    return () => {
      unsubscribe?.();
      finishHostActivity();
    };
  }, [enabled]);

  useEffect(
    () => () => {
      startupJob.current?.finish();
      buildJob.current?.finish();
      hostActivityJob.current?.finish();
      if (hostActivityTimer.current !== null) window.clearTimeout(hostActivityTimer.current);
    },
    [],
  );
}
