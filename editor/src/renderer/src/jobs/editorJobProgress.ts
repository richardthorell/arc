import { useSyncExternalStore } from 'react';

export type EditorJobProgress = {
  completed: number;
  total: number;
};

type EditorJobToken = {
  finish: () => void;
};

const listeners = new Set<() => void>();
let nextJobId = 1;
const activeJobs = new Set<number>();
let completedJobs = 0;
let totalJobs = 0;
let snapshot: EditorJobProgress | null = null;

const publish = () => {
  snapshot = activeJobs.size > 0 ? { completed: completedJobs, total: totalJobs } : null;
  listeners.forEach((listener) => listener());
};

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

export const getEditorJobProgress = () => snapshot;

export function beginEditorJob(): EditorJobToken {
  if (activeJobs.size === 0) {
    completedJobs = 0;
    totalJobs = 0;
  }

  const id = nextJobId++;
  activeJobs.add(id);
  totalJobs += 1;
  publish();

  let finished = false;
  return {
    finish: () => {
      if (finished) return;
      finished = true;
      if (!activeJobs.delete(id)) return;
      completedJobs += 1;
      publish();
    },
  };
}

export async function trackEditorJob<T>(operation: () => Promise<T>): Promise<T> {
  const job = beginEditorJob();
  try {
    return await operation();
  } finally {
    job.finish();
  }
}

export function useEditorJobProgress(): EditorJobProgress | null {
  return useSyncExternalStore(subscribe, getEditorJobProgress, getEditorJobProgress);
}
