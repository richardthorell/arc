import type { AiProviderAccountsSnapshot, AiProviderId } from './aiProviderTypes';

export type SourceControlFileState =
  'modified' | 'added' | 'deleted' | 'renamed' | 'copied' | 'untracked' | 'conflicted';

export type SourceControlFile = {
  path: string;
  indexState: SourceControlFileState | null;
  worktreeState: SourceControlFileState | null;
  originalPath?: string;
};

export type SourceControlSnapshot = {
  available: boolean;
  repositoryRoot: string;
  branch: string;
  detached: boolean;
  ahead: number;
  behind: number;
  files: SourceControlFile[];
  error: string;
};

export type SourceControlResult = {
  succeeded: boolean;
  output: string;
  error: string;
};

export type EditorHostPlatform = 'windows' | 'macos' | 'linux' | 'other';

export type EditorPathValidation = {
  valid: boolean;
  resolvedPath: string;
  message: string;
  source: 'configured' | 'environment' | 'default' | 'derived' | 'unresolved';
};

export type EditorSettingsSnapshot = {
  revision: number;
  values: Record<string, unknown>;
  sources: Record<string, 'default' | 'user' | 'project'>;
  restartRequired: string[];
  schema: EditorSettingDescriptor[];
  hostPlatform?: EditorHostPlatform;
  aiProviders?: AiProviderAccountsSnapshot;
  pathValidation?: Record<string, EditorPathValidation>;
};

export type EditorSettingDescriptor = {
  key: string;
  section:
    | 'Editor'
    | 'Renderer'
    | 'Input'
    | 'Cache'
    | 'Paths & Tools'
    | 'Extensions'
    | 'Source Control'
    | 'Recovery'
    | 'Build Tools'
    | 'Windows'
    | 'Android'
    | 'Linux'
    | 'Apple'
    | 'macOS'
    | 'iOS'
    | 'tvOS'
    | 'visionOS'
    | 'Web'
    | 'Xbox'
    | 'PlayStation'
    | 'Nintendo Switch'
    | 'OpenAI'
    | 'Anthropic';
  label: string;
  description: string;
  type: 'boolean' | 'number' | 'string' | 'enum';
  format?: 'color' | 'secret' | 'path';
  secretProvider?: AiProviderId;
  defaultValue: boolean | number | string;
  minimum?: number;
  maximum?: number;
  step?: number;
  options?: string[];
  optionLabels?: Record<string, string>;
  scopes: Array<'user' | 'project'>;
  restartRequired?: boolean;
  readOnly?: boolean;
  hostPlatforms?: EditorHostPlatform[];
  browsePath?: boolean;
};

export type ProjectTextFile = {
  path: string;
  text: string;
  modifiedAt: string;
};

export type RecoveryGeneration = {
  id: string;
  projectGuid: string;
  documentGuid: string;
  documentName: string;
  originalPath: string;
  recoveryPath: string;
  createdAt: string;
  historyRevision: number;
  sceneRevision: number;
  size: number;
};

export type RecoverySnapshot = {
  projectGuid: string;
  uncleanShutdown: boolean;
  heartbeatAt: string;
  generations: RecoveryGeneration[];
  totalBytes: number;
  error: string;
};
