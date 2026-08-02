export const arcProjectFormat = 'arc-project';
export const arcProjectFormatVersion = 2;

export type ArcProjectDependency = {
  kind: 'engine' | 'project' | 'plugin';
  id: string;
  version: string;
};

export type ArcProjectModuleDescriptor = {
  id: string;
  kind: 'editor' | 'runtime' | 'server';
  target: string;
  sourceRoot: string;
  enabled: boolean;
  dependencies: ArcProjectDependency[];
};

export type ArcProjectAssetReference = {
  guid: string;
  expectedType: string;
  pathHint: string;
};

export type ArcProjectDescriptor = {
  format: typeof arcProjectFormat;
  formatVersion: number;
  guid: string;
  name: string;
  engineVersion: string;
  paths: {
    source: string;
    content: string;
    config: string;
    plugins: string;
    saved: string;
    intermediate: string;
    build: string;
  };
  assetRoots: string[];
  modules: ArcProjectModuleDescriptor[];
  plugins: Array<{
    id: string;
    version: string;
    origin: string;
    required: boolean;
    enabled: boolean;
    path?: string;
  }>;
  defaultScene: ArcProjectAssetReference | null;
  startupScenes: ArcProjectAssetReference[];
  targetPlatforms: Array<{ id: string; enabled: boolean }>;
  toolchain: {
    compiler: string;
    minimumVersion: string;
    generator: string;
    architecture: string;
    cppStandard: number;
  };
  buildConfigurations: string[];
  renderer: { backend: 'none' | 'vulkan'; api: string; quality: string };
  cookProfiles: Array<{
    id: string;
    platform: string;
    architecture: string;
    renderer: string;
    api: string;
    textureFamily: string;
    configuration: string;
  }>;
  package: { applicationName: string; companyName: string; output: string; regionChunks: boolean };
  settings: { editor: string; renderer: string; input: string };
};

export type ArcProjectTemplate = {
  id: 'blank-3d' | 'blank-headless' | 'rendering-sample' | 'empty-cpp' | string;
  name: string;
  description: string;
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
  installationId: string;
  version: string;
  manifestPath: string;
  root: string;
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
  templates: ArcProjectTemplate[];
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
  template?: ArcProjectTemplate['id'];
};

export type ArcCloneProjectRequest = { source: string; destination: string };
