export type AgentPrefabCreateRequest = {
  rootGuid: string;
  path: string;
};

export type AgentPrefabInstantiateRequest = {
  path: string;
  parentGuid?: string;
};

export type AgentPrefabOperation =
  | { action: 'createPrefab'; value: AgentPrefabCreateRequest }
  | { action: 'instantiatePrefab'; value: AgentPrefabInstantiateRequest };

const requireNonEmpty = (value: unknown, name: string): string => {
  if (typeof value !== 'string' || value.trim() === '') throw new Error(`${name} must be a non-empty string`);
  return value;
};

export const requireAgentPrefabPath = (value: unknown): string => {
  const path = requireNonEmpty(value, 'value.path');
  const segments = path.split('/');
  if (
    path !== path.trim() ||
    path.includes('\\') ||
    path.startsWith('/') ||
    /^[A-Za-z]:/.test(path) ||
    segments.some((segment) => segment === '' || segment === '.' || segment === '..')
  ) {
    throw new Error('value.path must be a normalized content-relative path');
  }
  if (!path.toLowerCase().endsWith('.arcprefab')) throw new Error('value.path must reference a .arcprefab asset');
  return path;
};

export const parseAgentPrefabOperation = (action: string, input: unknown): AgentPrefabOperation => {
  const value = input && typeof input === 'object' && !Array.isArray(input) ? (input as Record<string, unknown>) : {};
  if (action === 'createPrefab') {
    return {
      action,
      value: {
        rootGuid: requireNonEmpty(value.rootGuid, 'value.rootGuid'),
        path: requireAgentPrefabPath(value.path),
      },
    };
  }
  if (action === 'instantiatePrefab') {
    const parentGuid =
      value.parentGuid === undefined ? undefined : requireNonEmpty(value.parentGuid, 'value.parentGuid');
    return {
      action,
      value: {
        path: requireAgentPrefabPath(value.path),
        ...(parentGuid ? { parentGuid } : {}),
      },
    };
  }
  throw new Error(`Unsupported prefab edit action: ${action}`);
};
