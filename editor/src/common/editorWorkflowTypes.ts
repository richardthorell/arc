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

export type EditorSettingsSnapshot = {
  revision: number;
  values: Record<string, unknown>;
  sources: Record<string, 'default' | 'user' | 'project'>;
  restartRequired: string[];
  schema: EditorSettingDescriptor[];
};

export type EditorSettingDescriptor = {
  key: string;
  section: 'Editor' | 'Renderer' | 'Input' | 'Cache' | 'Paths & Tools' | 'Extensions' | 'Source Control' | 'Recovery';
  label: string;
  description: string;
  type: 'boolean' | 'number' | 'string' | 'enum';
  defaultValue: boolean | number | string;
  minimum?: number;
  maximum?: number;
  step?: number;
  options?: string[];
  scopes: Array<'user' | 'project'>;
  restartRequired?: boolean;
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
