import type { ProjectSettingsSnapshot } from '../../../common/projectSettingsTypes';

// Only authoritative snapshots are cached; dialog drafts never enter this store.
let snapshot: ProjectSettingsSnapshot | undefined;
let epoch = 0;
let api: typeof window.arc.settings.project | undefined;

export function clearProjectSettingsCache() {
  ++epoch;
  snapshot = undefined;
  api = undefined;
}

export function cachedProjectSettings() {
  return api === window.arc?.settings?.project ? snapshot : undefined;
}

export function cacheProjectSettings(value: ProjectSettingsSnapshot) {
  api = window.arc?.settings?.project;
  snapshot = value;
}

export async function refreshProjectSettings() {
  const currentApi = window.arc?.settings?.project;
  if (!currentApi) throw new Error('Project Settings requires an updated editor host');
  const token = epoch;
  const value = await currentApi.read();
  if (token !== epoch || currentApi !== window.arc?.settings?.project)
    throw new Error('Project changed while reading settings');
  cacheProjectSettings(value);
  return value;
}
