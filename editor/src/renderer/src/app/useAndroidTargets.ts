import { useCallback, useEffect, useRef, useState } from 'react';

import type { AndroidTargetReference, AndroidTargetSnapshot } from '../../../common/androidDeviceTypes';

export const androidPreferenceKey = (project: string, sdkRoot: string): string =>
  `arc.android-target.v1:${JSON.stringify([project, sdkRoot])}`;

/** Workbench-owned selection. A missing preferred identity is retained, never substituted. */
export function useAndroidTargets(active: boolean, project: string) {
  const [snapshot, setSnapshot] = useState<AndroidTargetSnapshot | null>(null);
  const [selected, setSelected] = useState<AndroidTargetReference | null>(null);
  const [error, setError] = useState('');
  const [starting, setStarting] = useState(false);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const generation = useRef(0);
  const selectionContext = useRef('');
  const selectedId = useRef('');
  const refresh = useCallback(() => setRefreshVersion((value) => value + 1), []);

  useEffect(() => {
    const epoch = ++generation.current;
    setSnapshot(null);
    setError('');
    setStarting(false);
    if (!active) return;
    let busy = false;
    const poll = async () => {
      if (busy) return;
      busy = true;
      try {
        if (!window.arcAndroid) throw new Error('Android target service is unavailable');
        const result = await window.arcAndroid.listTargets();
        if (generation.current !== epoch) return;
        const key = androidPreferenceKey(project, result.sdkRoot);
        if (selectionContext.current !== key) {
          let saved: string | null = null;
          try {
            saved = window.localStorage.getItem(key);
          } catch {
            /* Selection still works without persistence. */
          }
          const id = saved ?? '';
          selectedId.current = id;
          selectionContext.current = key;
          setSelected(id ? { sdkRoot: result.sdkRoot, id } : null);
        }
        // Keep looking when discovery initially found no ready device. Never replace
        // a remembered selection, even if it has temporarily disconnected.
        if (!selectedId.current) {
          const ready = result.targets.filter((target) => target.ready);
          if (ready.length === 1) {
            const id = ready[0].id;
            selectedId.current = id;
            setSelected({ sdkRoot: result.sdkRoot, id });
            try {
              window.localStorage.setItem(key, id);
            } catch {
              /* Machine-local persistence is best effort. */
            }
          }
        }
        setSnapshot(result);
        setError('');
      } catch (reason) {
        if (generation.current === epoch) {
          setSnapshot(null);
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      } finally {
        busy = false;
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 2000);
    return () => {
      generation.current = epoch + 1;
      window.clearInterval(timer);
    };
  }, [active, project, refreshVersion]);

  const select = (id: string) => {
    const target = snapshot?.targets.find((entry) => entry.id === id);
    if (!snapshot || !target || (!target.ready && !target.canStart && target.state !== 'starting')) return;
    setSelected({ sdkRoot: snapshot.sdkRoot, id });
    selectedId.current = id;
    const key = androidPreferenceKey(project, snapshot.sdkRoot);
    selectionContext.current = key;
    try {
      window.localStorage.setItem(key, id);
    } catch {
      /* Keep the in-memory choice. */
    }
  };
  const start = async () => {
    if (!selected || !snapshot || selected.sdkRoot !== snapshot.sdkRoot || starting) return;
    const epoch = generation.current;
    setStarting(true);
    setError('');
    try {
      if (!window.arcAndroid) throw new Error('Android target service is unavailable');
      await window.arcAndroid.startEmulator(selected);
      if (epoch === generation.current) refresh();
    } catch (reason) {
      if (epoch === generation.current) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (epoch === generation.current) setStarting(false);
    }
  };
  return { snapshot, selected, error, starting, select, start, refresh };
}

export type AndroidTargetController = ReturnType<typeof useAndroidTargets>;
