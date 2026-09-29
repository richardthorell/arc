export type ProjectReloadAction = 'hot-reload' | 'restart-play-session' | 'restart-native-host';

export type ProjectReloadChange =
  | 'additive-schema'
  | 'rename-only'
  | 'field-wire-kind'
  | 'removed-component'
  | 'schema-downgrade'
  | 'identity-change'
  | 'abi-change';

export interface ProjectReloadDecision {
  action: ProjectReloadAction;
  reasons: ProjectReloadChange[];
}

const nativeHostRestartChanges = new Set<ProjectReloadChange>([
  'removed-component',
  'schema-downgrade',
  'identity-change',
  'abi-change',
]);

/**
 * Classifies a staged project-module generation before it is published.
 *
 * The most restrictive detected change wins. Compatible additive/rename-only
 * changes can stay in the current Play world, field wire-kind changes require
 * a clean Play-session restart, and changes that invalidate the module/schema
 * boundary require the native host to restart.
 */
export const classifyProjectReload = (changes: readonly ProjectReloadChange[]): ProjectReloadDecision => {
  const reasons = [...new Set(changes)];

  if (reasons.some((change) => nativeHostRestartChanges.has(change))) {
    return { action: 'restart-native-host', reasons };
  }

  if (reasons.includes('field-wire-kind')) {
    return { action: 'restart-play-session', reasons };
  }

  return { action: 'hot-reload', reasons };
};
