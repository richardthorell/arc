import { useSyncExternalStore } from 'react';

export type EditorJobPriority = 'background' | 'normal' | 'foreground';

export type EditorJobProgress = {
  label: string;
  completed: number;
  total: number;
  indeterminate: boolean;
};

export type EditorJobUpdate = {
  label?: string;
  completed?: number;
  total?: number;
  indeterminate?: boolean;
};

export type EditorJobOptions = EditorJobUpdate & {
  priority?: EditorJobPriority;
};

export type EditorJobToken = {
  update: (update: EditorJobUpdate) => void;
  finish: () => void;
};

type EditorJobRecord = {
  id: number;
  label: string;
  completed: number;
  total: number;
  indeterminate: boolean;
  priority: EditorJobPriority;
  finished: boolean;
  explicitProgress: boolean;
};

const listeners = new Set<() => void>();
let nextJobId = 1;
const jobs = new Map<number, EditorJobRecord>();
let snapshot: EditorJobProgress | null = null;

const priorityValue = (priority: EditorJobPriority) => {
  if (priority === 'foreground') return 2;
  if (priority === 'normal') return 1;
  return 0;
};

const publish = () => {
  const active = [...jobs.values()].filter((job) => !job.finished);
  if (active.length === 0) {
    jobs.clear();
    snapshot = null;
    listeners.forEach((listener) => listener());
    return;
  }

  const highestPriority = Math.max(...active.map((job) => priorityValue(job.priority)));
  const activeAtPriority = active.filter((job) => priorityValue(job.priority) === highestPriority);
  const lead = activeAtPriority[0];

  if (lead.explicitProgress) {
    snapshot = {
      label: lead.label,
      completed: lead.completed,
      total: lead.total,
      indeterminate: lead.indeterminate,
    };
  } else {
    const cohort = [...jobs.values()].filter((job) => priorityValue(job.priority) === highestPriority);
    snapshot = {
      label: lead.label,
      completed: cohort.filter((job) => job.finished).length,
      total: cohort.length,
      indeterminate: false,
    };
  }

  listeners.forEach((listener) => listener());
};

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

export const getEditorJobProgress = () => snapshot;

export const beginEditorJob = (label = 'Working', options: EditorJobOptions = {}): EditorJobToken => {
  const id = nextJobId++;
  const explicitProgress =
    options.completed !== undefined || options.total !== undefined || options.indeterminate !== undefined;
  const record: EditorJobRecord = {
    id,
    label,
    completed: options.completed ?? 0,
    total: options.total ?? 0,
    indeterminate: options.indeterminate ?? false,
    priority: options.priority ?? 'normal',
    finished: false,
    explicitProgress,
  };
  jobs.set(id, record);
  publish();

  return {
    update: (update) => {
      const current = jobs.get(id);
      if (!current || current.finished) return;
      if (update.label !== undefined) current.label = update.label;
      if (update.completed !== undefined) current.completed = update.completed;
      if (update.total !== undefined) current.total = update.total;
      if (update.indeterminate !== undefined) current.indeterminate = update.indeterminate;
      if (update.completed !== undefined || update.total !== undefined || update.indeterminate !== undefined)
        current.explicitProgress = true;
      publish();
    },
    finish: () => {
      const current = jobs.get(id);
      if (!current || current.finished) return;
      current.finished = true;
      publish();
    },
  };
};

export const trackEditorJob = async <T>(
  label: string,
  operation: () => Promise<T>,
  options: EditorJobOptions = {},
): Promise<T> => {
  const job = beginEditorJob(label, options);
  try {
    return await operation();
  } finally {
    job.finish();
  }
};

export const useEditorJobProgress = () => useSyncExternalStore(subscribe, getEditorJobProgress, getEditorJobProgress);
