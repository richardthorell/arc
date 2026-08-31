import { useEffect, useRef } from 'react';

import type { ArcBuildSnapshot } from '../../../common/buildTypes';
import type { StartupState } from '../app/workbenchTypes';
import { beginEditorJob, type EditorJobToken } from './editorJobProgress';

const buildActivityLabel = (snapshot: ArcBuildSnapshot) => {
  if (snapshot.state === 'configuring') return 'Configuring project';
  if (snapshot.state === 'building') return 'Building project';
  if (snapshot.state === 'cleaning') return 'Cleaning project';
  return '';
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

export function useEditorActivityProgress(startupState: StartupState | null, activeScene?: string, enabled = true) {
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
      const label = snapshot ? buildActivityLabel(snapshot) : '';
      if (!label) {
        buildJob.current?.finish();
        buildJob.current = null;
        return;
      }

      const update = { label, completed: 0, total: 0, indeterminate: true };
      if (!buildJob.current) buildJob.current = beginEditorJob(label, { priority: 'foreground', ...update });
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
