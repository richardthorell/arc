export const arcProjectFormat = 'arc-project';
export const arcProjectFormatVersion = 1;

export type ArcProjectDescriptor = {
  format: typeof arcProjectFormat;
  formatVersion: typeof arcProjectFormatVersion;
  guid: string;
  name: string;
  engineVersion: string;
  assetRoots: string[];
  startupScenes: string[];
  modules: string[];
  extensions: string[];
  settings: {
    editor: string;
    renderer: string;
    input: string;
  };
};

export type ArcRecentProject = {
  descriptorPath: string;
  projectRoot: string;
  guid: string;
  name: string;
  engineVersion: string;
  lastOpenedAt: string;
  missing: boolean;
};

export type ArcEngineInstallation = {
  version: string;
  editorPath: string;
  current: boolean;
};

export type ArcProjectCompatibility = 'compatible' | 'upgradeRequired' | 'newerEngineRequired';

export type ArcProjectCandidate = {
  descriptor: ArcProjectDescriptor;
  descriptorPath: string;
  projectRoot: string;
  compatibility: ArcProjectCompatibility;
  writable: boolean;
  diagnostics: string[];
};

export type ArcProjectBrowserSnapshot = {
  currentEngineVersion: string;
  activeProject: ArcProjectCandidate | null;
  recentProjects: ArcRecentProject[];
  installations: ArcEngineInstallation[];
  hostConnected: boolean;
  hostError: string;
};

export type ArcProjectOperationResult = {
  succeeded: boolean;
  error?: string;
  project?: ArcProjectCandidate;
};

export type ArcCreateProjectRequest = {
  name: string;
  destination: string;
  template?: 'empty' | 'mountain';
};

export type ArcCloneProjectRequest = {
  source: string;
  destination: string;
};
