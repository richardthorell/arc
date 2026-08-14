import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

import {
  arcProjectFormat,
  arcProjectFormatVersion,
  type ArcCloneProjectRequest,
  type ArcCreateProjectRequest,
  type ArcEngineInstallation,
  type ArcProjectBrowserSnapshot,
  type ArcProjectCandidate,
  type ArcProjectDescriptor,
  type ArcProjectOperationResult,
  type ArcRecentProject,
  type ArcProjectTemplate,
} from '../common/projectTypes';

type ProjectHost = {
  connected: boolean;
  error: string;
  command(
    type: string,
    payload?: Record<string, unknown>,
  ): Promise<{
    succeeded: boolean;
    error?: string;
  }>;
};

const descriptorExtension = '.arcproject';
const recentFileName = 'recent-projects.v1.json';

const resolveDevelopmentProjectTool = (): string => {
  const executable = process.platform === 'win32' ? 'arc-project.exe' : 'arc-project';
  const repository = path.resolve(process.cwd(), '..');
  for (const preset of ['editor-vulkan', 'default', 'editor-no-vulkan'])
    for (const configuration of ['RelWithDebInfo', 'Release', 'Debug', '']) {
      const candidate = path.join(
        repository,
        'out',
        'build',
        preset,
        'tools',
        'project_cli',
        configuration,
        executable,
      );
      if (fs.existsSync(candidate)) return candidate;
    }
  return '';
};

const writeJsonAtomic = (target: string, value: unknown): void => {
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const temporary = `${target}.tmp-${process.pid}-${Date.now()}`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  fs.renameSync(temporary, target);
};

const normalizeStringArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === 'string') : [];

const normalizeProjectRelativePath = (value: string, field: string): string => {
  const normalized = value.trim().replaceAll('\\', '/').replace(/^\/+/, '');
  if (
    !normalized ||
    normalized === '..' ||
    normalized.startsWith('../') ||
    path.posix.isAbsolute(normalized) ||
    /^[a-z]:/i.test(normalized)
  )
    throw new Error(`${field} must contain project-relative paths`);
  return path.posix.normalize(normalized);
};

const normalizeProjectRelativePaths = (value: unknown, field: string): string[] =>
  normalizeStringArray(value).map((entry) => normalizeProjectRelativePath(entry, field));

const objectValue = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

const stringValue = (value: unknown, fallback = ''): string => (typeof value === 'string' ? value : fallback);

const assetReference = (value: unknown): ArcProjectDescriptor['defaultScene'] => {
  const source = objectValue(value);
  const guid = stringValue(source.guid);
  const pathHint = stringValue(source.pathHint);
  if (!guid && !pathHint) return null;
  return { guid, expectedType: stringValue(source.expectedType, 'scene'), pathHint };
};

const parseDescriptor = (value: unknown): ArcProjectDescriptor => {
  if (!value || typeof value !== 'object') throw new Error('Project descriptor must be a JSON object');
  const source = value as Partial<ArcProjectDescriptor>;
  if (source.format !== arcProjectFormat) throw new Error(`Expected project format '${arcProjectFormat}'`);
  if (source.formatVersion !== 1 && source.formatVersion !== arcProjectFormatVersion)
    throw new Error(`Unsupported project format version ${String(source.formatVersion)}`);
  if (
    typeof source.guid !== 'string' ||
    !/^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i.test(source.guid)
  )
    throw new Error('Project GUID is missing or malformed');
  if (typeof source.name !== 'string' || !source.name.trim()) throw new Error('Project name is required');
  if (typeof source.engineVersion !== 'string' || !source.engineVersion.trim())
    throw new Error('Project engine version is required');
  const raw = value as Record<string, unknown>;
  const paths = objectValue(raw.paths);
  const legacyModules = normalizeStringArray(raw.modules);
  const modules: ArcProjectDescriptor['modules'] = Array.isArray(raw.modules)
    ? raw.modules.flatMap((entry, index) => {
        if (typeof entry === 'string')
          return [
            {
              id: entry,
              kind: 'runtime' as const,
              target: entry,
              sourceRoot: 'Source',
              enabled: true,
              dependencies: [],
            },
          ];
        const module = objectValue(entry);
        const kind = module.kind === 'editor' || module.kind === 'server' ? module.kind : 'runtime';
        return [
          {
            id: stringValue(module.id, legacyModules[index] ?? ''),
            kind,
            target: stringValue(module.target),
            sourceRoot: stringValue(module.sourceRoot, 'Source'),
            enabled: module.enabled !== false,
            dependencies: Array.isArray(module.dependencies)
              ? module.dependencies.map((dependency) => {
                  const item = objectValue(dependency);
                  const dependencyKind: 'engine' | 'project' | 'plugin' =
                    item.kind === 'project' || item.kind === 'plugin' ? item.kind : 'engine';
                  return { kind: dependencyKind, id: stringValue(item.id), version: stringValue(item.version) };
                })
              : [],
          },
        ];
      })
    : [];
  const startupScenes = Array.isArray(raw.startupScenes)
    ? raw.startupScenes.flatMap((entry) => {
        if (typeof entry === 'string') return [{ guid: '', expectedType: 'scene', pathHint: entry }];
        const reference = assetReference(entry);
        return reference ? [reference] : [];
      })
    : [];
  const settings = objectValue(raw.settings);
  const renderer = objectValue(raw.renderer);
  const toolchain = objectValue(raw.toolchain);
  const packageSettings = objectValue(raw.package);
  return {
    format: arcProjectFormat,
    formatVersion: arcProjectFormatVersion,
    guid: source.guid,
    name: source.name.trim(),
    engineVersion: source.engineVersion,
    paths: {
      source: normalizeProjectRelativePath(stringValue(paths.source, 'Source'), 'paths.source'),
      content: normalizeProjectRelativePath(
        stringValue(paths.content, source.formatVersion === 1 ? 'assets' : 'Content'),
        'paths.content',
      ),
      config: normalizeProjectRelativePath(
        stringValue(paths.config, source.formatVersion === 1 ? 'config' : 'Config'),
        'paths.config',
      ),
      plugins: normalizeProjectRelativePath(stringValue(paths.plugins, 'Plugins'), 'paths.plugins'),
      saved: normalizeProjectRelativePath(stringValue(paths.saved, 'Saved'), 'paths.saved'),
      intermediate: normalizeProjectRelativePath(stringValue(paths.intermediate, 'Intermediate'), 'paths.intermediate'),
      build: normalizeProjectRelativePath(stringValue(paths.build, 'Build'), 'paths.build'),
    },
    assetRoots: normalizeStringArray(source.assetRoots).length
      ? normalizeProjectRelativePaths(source.assetRoots, 'assetRoots')
      : ['Content'],
    modules,
    plugins: Array.isArray(raw.plugins)
      ? raw.plugins.map((entry) => {
          const plugin = objectValue(entry);
          return {
            id: stringValue(plugin.id),
            version: stringValue(plugin.version),
            origin: stringValue(plugin.origin, 'engine'),
            required: plugin.required !== false,
            enabled: plugin.enabled !== false,
            path: typeof plugin.path === 'string' ? plugin.path : undefined,
          };
        })
      : [],
    defaultScene: assetReference(raw.defaultScene) ?? startupScenes[0] ?? null,
    startupScenes,
    targetPlatforms: Array.isArray(raw.targetPlatforms)
      ? raw.targetPlatforms.map((entry) => {
          const platform = objectValue(entry);
          return { id: stringValue(platform.id), enabled: platform.enabled !== false };
        })
      : [],
    toolchain: {
      compiler: stringValue(toolchain.compiler, 'auto'),
      minimumVersion: stringValue(toolchain.minimumVersion),
      generator: stringValue(toolchain.generator, 'auto'),
      architecture: stringValue(toolchain.architecture, 'x86_64'),
      cppStandard: typeof toolchain.cppStandard === 'number' ? toolchain.cppStandard : 20,
    },
    buildConfigurations: normalizeStringArray(raw.buildConfigurations).length
      ? normalizeStringArray(raw.buildConfigurations)
      : ['Debug', 'RelWithDebInfo', 'Shipping'],
    renderer: {
      backend: renderer.backend === 'none' ? 'none' : 'vulkan',
      api: stringValue(renderer.api, renderer.backend === 'none' ? '' : '1.2'),
      quality: stringValue(renderer.quality, 'standard'),
    },
    cookProfiles: Array.isArray(raw.cookProfiles)
      ? raw.cookProfiles.map((entry) => {
          const profile = objectValue(entry);
          return {
            id: stringValue(profile.id),
            platform: stringValue(profile.platform),
            architecture: stringValue(profile.architecture, 'x86_64'),
            renderer: stringValue(profile.renderer, 'vulkan'),
            api: stringValue(profile.api, '1.2'),
            textureFamily: stringValue(profile.textureFamily, 'bc'),
            configuration: stringValue(profile.configuration, 'Shipping'),
          };
        })
      : [],
    package: {
      applicationName: stringValue(packageSettings.applicationName, source.name.trim()),
      companyName: stringValue(packageSettings.companyName),
      output: stringValue(packageSettings.output, 'Build/Packages'),
      regionChunks: packageSettings.regionChunks !== false,
    },
    settings: {
      editor: normalizeProjectRelativePath(stringValue(settings.editor, 'Config/Editor.json'), 'settings.editor'),
      renderer: normalizeProjectRelativePath(
        stringValue(settings.renderer, 'Config/Renderer.json'),
        'settings.renderer',
      ),
      input: normalizeProjectRelativePath(stringValue(settings.input, 'Config/Input.json'), 'settings.input'),
    },
  };
};

const resolveDescriptorPath = (candidate: string): string => {
  const resolved = path.resolve(candidate);
  if (fs.existsSync(resolved) && fs.statSync(resolved).isFile()) return resolved;
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isDirectory())
    throw new Error(`Project path does not exist: ${resolved}`);
  const descriptors = fs
    .readdirSync(resolved, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(descriptorExtension))
    .map((entry) => path.join(resolved, entry.name));
  if (descriptors.length !== 1)
    throw new Error(
      descriptors.length === 0
        ? `No ${descriptorExtension} descriptor exists in ${resolved}`
        : `More than one ${descriptorExtension} descriptor exists in ${resolved}`,
    );
  return descriptors[0];
};

const compareVersion = (left: string, right: string): number => {
  const parts = (value: string) =>
    value
      .replace(/^v/i, '')
      .split(/[.+-]/)
      .slice(0, 3)
      .map((part) => Number.parseInt(part, 10) || 0);
  const a = parts(left);
  const b = parts(right);
  for (let index = 0; index < 3; ++index) {
    if (a[index] !== b[index]) return a[index] < b[index] ? -1 : 1;
  }
  return 0;
};

export class ProjectService {
  private readonly recentPath: string;
  private readonly currentEngineVersion: string;
  private readonly currentEditorPath: string;
  private readonly projectToolPath: string;
  private readonly templatesRoot: string;
  private readonly builtinAssetsRoot: string;
  private readonly installationRegistryPath?: string;
  private readonly host: ProjectHost;
  private activeProject: ArcProjectCandidate | null = null;

  constructor(options: {
    userDataPath: string;
    currentEngineVersion: string;
    currentEditorPath: string;
    projectToolPath?: string;
    templatesRoot?: string;
    builtinAssetsRoot?: string;
    installationRegistryPath?: string;
    host: ProjectHost;
  }) {
    this.recentPath = path.join(options.userDataPath, recentFileName);
    this.currentEngineVersion = options.currentEngineVersion;
    this.currentEditorPath = options.currentEditorPath;
    this.projectToolPath =
      options.projectToolPath ?? process.env.ARC_PROJECT_TOOL_PATH ?? resolveDevelopmentProjectTool();
    this.templatesRoot = options.templatesRoot ?? path.resolve(process.cwd(), '..', 'templates');
    this.builtinAssetsRoot = options.builtinAssetsRoot ?? '';
    this.installationRegistryPath = options.installationRegistryPath;
    this.host = options.host;
  }

  snapshot(): ArcProjectBrowserSnapshot {
    return {
      currentEngineVersion: this.currentEngineVersion,
      activeProject: this.activeProject,
      recentProjects: this.readRecents(),
      installations: this.engineInstallations(),
      templates: this.projectTemplates(),
      hostConnected: this.host.connected,
      hostError: this.host.error,
    };
  }

  active(): ArcProjectCandidate | null {
    return this.activeProject;
  }

  projectTool(): string {
    return this.projectToolPath;
  }

  inspect(candidate: string): ArcProjectCandidate {
    const descriptorPath = resolveDescriptorPath(candidate);
    const rawDescriptor = JSON.parse(fs.readFileSync(descriptorPath, 'utf8')) as unknown;
    const sourceFormatVersion = objectValue(rawDescriptor).formatVersion;
    const descriptor = parseDescriptor(rawDescriptor);
    if (sourceFormatVersion === arcProjectFormatVersion)
      this.runProjectTool(['validate', '--project', descriptorPath, '--require-paths']);
    const comparison = compareVersion(descriptor.engineVersion, this.currentEngineVersion);
    return {
      descriptor,
      descriptorPath,
      projectRoot: path.dirname(descriptorPath),
      compatibility: comparison === 0 ? 'compatible' : comparison < 0 ? 'upgradeRequired' : 'newerEngineRequired',
      writable: comparison === 0,
      diagnostics:
        comparison === 0
          ? []
          : [
              comparison < 0
                ? `Project requires an explicit upgrade from ${descriptor.engineVersion} to ${this.currentEngineVersion}`
                : `Project requires ARC ${descriptor.engineVersion}; the running editor is ${this.currentEngineVersion}`,
            ],
    };
  }

  async open(
    candidate: string,
    options: { readOnly?: boolean; upgrade?: boolean } = {},
  ): Promise<ArcProjectOperationResult> {
    try {
      let project = this.inspect(candidate);
      if (project.compatibility === 'upgradeRequired' && options.upgrade) project = this.upgrade(project);
      if (project.compatibility !== 'compatible' && !options.readOnly)
        return { succeeded: false, error: project.diagnostics[0], project };
      if (!this.host.connected)
        return { succeeded: false, error: this.host.error || 'Native editor host is unavailable' };
      const editorModule = this.resolveEditorModule(project);
      const response = await this.host.command('project.open', {
        name: project.descriptor.name,
        root: project.projectRoot,
        descriptorPath: project.descriptorPath,
        contentRoots: project.descriptor.assetRoots.map((entry) => path.join(project.projectRoot, entry)),
        builtinContentRoots: this.builtinAssetsRoot ? [this.builtinAssetsRoot] : [],
        cacheRoot: path.join(project.projectRoot, project.descriptor.paths.intermediate, 'Cache'),
        defaultScene: project.descriptor.defaultScene?.pathHint ?? '',
        projectGuid: project.descriptor.guid,
        engineVersion: project.descriptor.engineVersion,
        editorModuleId: editorModule?.id ?? '',
        editorModulePath: editorModule?.path ?? '',
        readOnly: options.readOnly || project.compatibility !== 'compatible',
      });
      if (!response.succeeded) return { succeeded: false, error: response.error || 'Native host rejected the project' };
      this.activeProject = { ...project, writable: !options.readOnly && project.compatibility === 'compatible' };
      this.touchRecent(this.activeProject);
      return { succeeded: true, project: this.activeProject };
    } catch (error) {
      return { succeeded: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  async close(): Promise<ArcProjectOperationResult> {
    if (this.host.connected) {
      const response = await this.host.command('project.close');
      if (!response.succeeded) return { succeeded: false, error: response.error || 'Could not close project' };
    }
    this.activeProject = null;
    return { succeeded: true };
  }

  launchMatchingEngine(candidate: string): ArcProjectOperationResult {
    try {
      const project = this.inspect(candidate);
      const installation = this.engineInstallations().find(
        (entry) =>
          entry.version === project.descriptor.engineVersion && entry.editorPath && fs.existsSync(entry.editorPath),
      );
      if (!installation) throw new Error(`No registered ARC ${project.descriptor.engineVersion} editor is available`);
      const child = spawn(installation.editorPath, [project.descriptorPath], {
        detached: true,
        shell: false,
        windowsHide: false,
        stdio: 'ignore',
      });
      child.unref();
      return { succeeded: true, project };
    } catch (error) {
      return { succeeded: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  create(request: ArcCreateProjectRequest): ArcProjectOperationResult {
    try {
      const safeName = request.name.trim();
      if (!safeName) throw new Error('Project name is required');
      const template =
        request.template === 'mountain'
          ? 'rendering-sample'
          : request.template === 'empty'
            ? 'empty-cpp'
            : (request.template ?? 'blank-3d');
      this.runProjectTool([
        'create',
        '--name',
        safeName,
        '--destination',
        path.resolve(request.destination),
        '--template',
        template,
        '--templates',
        this.templatesRoot,
        '--engine',
        this.currentEngineVersion,
      ]);
      return { succeeded: true, project: this.inspect(request.destination) };
    } catch (error) {
      return { succeeded: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  async openOrCreateQuickStartProject(destination: string): Promise<ArcProjectOperationResult> {
    const projectRoot = path.resolve(destination);
    try {
      let descriptorPath = '';
      const needsCreation =
        !fs.existsSync(projectRoot) ||
        (fs.statSync(projectRoot).isDirectory() && fs.readdirSync(projectRoot).length === 0);
      if (needsCreation) {
        const created = this.create({
          name: 'ARC Editor Development',
          destination: projectRoot,
          template: 'blank-3d',
        });
        if (!created.succeeded || !created.project) return created;
        descriptorPath = created.project.descriptorPath;
      } else descriptorPath = resolveDescriptorPath(projectRoot);

      return await this.open(descriptorPath, { upgrade: true });
    } catch (error) {
      return { succeeded: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  async clone(request: ArcCloneProjectRequest): Promise<ArcProjectOperationResult> {
    let destination = '';
    let destinationPrepared = false;
    try {
      destination = path.resolve(request.destination);
      if (destination === path.parse(destination).root)
        throw new Error('Clone destination cannot be a filesystem root');
      if (fs.existsSync(destination) && fs.readdirSync(destination).length)
        throw new Error('Clone destination must be empty');
      const localSource = path.resolve(request.source);
      if (fs.existsSync(localSource) && fs.statSync(localSource).isDirectory()) {
        const relativeDestination = path.relative(localSource, destination);
        if (
          !relativeDestination ||
          (!relativeDestination.startsWith(`..${path.sep}`) &&
            relativeDestination !== '..' &&
            !path.isAbsolute(relativeDestination))
        )
          throw new Error('Clone destination cannot be the source or one of its descendants');
        if (fs.existsSync(destination)) fs.rmSync(destination, { recursive: true });
        fs.mkdirSync(path.dirname(destination), { recursive: true });
        destinationPrepared = true;
        fs.cpSync(localSource, destination, { recursive: true, errorOnExist: true });
      } else {
        fs.mkdirSync(path.dirname(destination), { recursive: true });
        destinationPrepared = true;
        await new Promise<void>((resolve, reject) => {
          const child = spawn('git', ['clone', '--', request.source, destination], {
            shell: false,
            windowsHide: true,
            stdio: ['ignore', 'pipe', 'pipe'],
          });
          let diagnostic = '';
          child.stderr.on('data', (chunk) => {
            diagnostic += String(chunk);
          });
          child.once('error', reject);
          child.once('exit', (code) =>
            code === 0 ? resolve() : reject(new Error(diagnostic.trim() || `git exited ${code}`)),
          );
        });
      }
      return { succeeded: true, project: this.inspect(destination) };
    } catch (error) {
      if (destinationPrepared && destination && fs.existsSync(destination)) {
        const resolvedDestination = path.resolve(destination);
        if (resolvedDestination === destination && resolvedDestination !== path.parse(resolvedDestination).root)
          fs.rmSync(resolvedDestination, { recursive: true, force: true });
      }
      return { succeeded: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  removeRecent(descriptorPath: string): void {
    this.writeRecents(
      this.readRecents().filter((entry) => path.resolve(entry.descriptorPath) !== path.resolve(descriptorPath)),
    );
  }

  async deleteProject(
    descriptorPath: string,
    moveToTrash: (projectRoot: string) => Promise<void>,
  ): Promise<ArcProjectOperationResult> {
    try {
      const resolvedDescriptor = path.resolve(descriptorPath);
      const recent = this.readRecents().find((entry) => path.resolve(entry.descriptorPath) === resolvedDescriptor);
      if (!recent) throw new Error('Only a project listed in Recent Projects can be moved to the trash');
      if (this.activeProject && path.resolve(this.activeProject.descriptorPath) === resolvedDescriptor)
        throw new Error('Close the active project before moving it to the trash');
      if (!fs.existsSync(resolvedDescriptor)) {
        this.removeRecent(resolvedDescriptor);
        return { succeeded: true };
      }

      const project = this.inspect(resolvedDescriptor);
      const projectRoot = path.resolve(project.projectRoot);
      if (projectRoot === path.parse(projectRoot).root)
        throw new Error('A filesystem root cannot be moved to the trash');
      if (fs.lstatSync(projectRoot).isSymbolicLink())
        throw new Error('Symbolic-link project roots cannot be moved to the trash');
      if (path.resolve(recent.projectRoot) !== projectRoot)
        throw new Error('The recent-project root no longer matches its descriptor');

      await moveToTrash(projectRoot);
      this.removeRecent(resolvedDescriptor);
      return { succeeded: true, project };
    } catch (error) {
      return { succeeded: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  private upgrade(project: ArcProjectCandidate): ArcProjectCandidate {
    if (project.compatibility !== 'upgradeRequired') return project;
    this.runProjectTool(['upgrade', '--project', project.descriptorPath, '--engine', this.currentEngineVersion]);
    return this.inspect(project.descriptorPath);
  }

  private engineInstallations(): ArcEngineInstallation[] {
    try {
      const args = ['engine', 'list'];
      if (this.installationRegistryPath) args.push('--registry', this.installationRegistryPath);
      const result = this.runProjectTool(args) as { installations?: Array<Record<string, string>> };
      const installations = (result.installations ?? []).map((entry) => ({
        installationId: entry.installationId ?? '',
        version: entry.engineVersion ?? '',
        manifestPath: entry.manifest ?? '',
        root: entry.root ?? '',
        editorPath: entry.editor ?? '',
        current:
          entry.engineVersion === this.currentEngineVersion &&
          path.resolve(entry.editor ?? '') === path.resolve(this.currentEditorPath),
      }));
      if (!installations.some((entry) => entry.current))
        installations.unshift({
          installationId: 'running-editor',
          version: this.currentEngineVersion,
          manifestPath: '',
          root: path.dirname(this.currentEditorPath),
          editorPath: this.currentEditorPath,
          current: true,
        });
      return installations;
    } catch {
      return [
        {
          installationId: 'running-editor',
          version: this.currentEngineVersion,
          manifestPath: '',
          root: path.dirname(this.currentEditorPath),
          editorPath: this.currentEditorPath,
          current: true,
        },
      ];
    }
  }

  private projectTemplates(): ArcProjectTemplate[] {
    try {
      return fs
        .readdirSync(this.templatesRoot, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .flatMap((entry) => {
          try {
            const manifest = JSON.parse(
              fs.readFileSync(path.join(this.templatesRoot, entry.name, 'template.arc-template.json'), 'utf8'),
            ) as Record<string, unknown>;
            return [
              {
                id: stringValue(manifest.id, entry.name),
                name: stringValue(manifest.name, entry.name),
                description: stringValue(manifest.description),
              },
            ];
          } catch {
            return [];
          }
        });
    } catch {
      return [];
    }
  }

  private resolveEditorModule(project: ArcProjectCandidate): { id: string; path: string } | null {
    if (!project.writable) return null;
    const selectionPath = path.join(project.projectRoot, project.descriptor.paths.saved, 'Editor', 'active-build.json');
    try {
      const selection = JSON.parse(fs.readFileSync(selectionPath, 'utf8')) as Record<string, unknown>;
      if (
        selection.format !== 'arc-active-build' ||
        selection.formatVersion !== 1 ||
        typeof selection.moduleManifest !== 'string'
      )
        return null;
      const manifestPath = path.resolve(project.projectRoot, selection.moduleManifest);
      const relativeManifest = path.relative(project.projectRoot, manifestPath);
      if (relativeManifest.startsWith(`..${path.sep}`) || path.isAbsolute(relativeManifest)) return null;
      const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as {
        projectGuid?: string;
        engineVersion?: string;
        modules?: Array<{ id?: string; path?: string }>;
      };
      if (
        manifest.projectGuid !== project.descriptor.guid ||
        manifest.engineVersion !== project.descriptor.engineVersion
      )
        return null;
      const configuredId = project.descriptor.modules.find((module) => module.kind === 'editor' && module.enabled)?.id;
      const module = manifest.modules?.find((entry) => entry.id === configuredId && typeof entry.path === 'string');
      if (!configuredId || !module?.path || !fs.existsSync(module.path)) return null;
      const modulePath = path.resolve(module.path);
      const buildRoot = path.resolve(project.projectRoot, project.descriptor.paths.build);
      const relativeModule = path.relative(buildRoot, modulePath);
      if (!relativeModule || relativeModule.startsWith(`..${path.sep}`) || path.isAbsolute(relativeModule)) return null;
      return { id: configuredId, path: modulePath };
    } catch {
      return null;
    }
  }

  private runProjectTool(arguments_: string[]): unknown {
    if (!this.projectToolPath || !fs.existsSync(this.projectToolPath))
      throw new Error('The ARC project generator is unavailable');
    const result = spawnSync(this.projectToolPath, [...arguments_, '--json'], {
      cwd: process.cwd(),
      encoding: 'utf8',
      shell: false,
      windowsHide: true,
    });
    const stdout = typeof result.stdout === 'string' ? result.stdout.trim() : '';
    const stderr = typeof result.stderr === 'string' ? result.stderr.trim() : '';
    let response: { succeeded?: boolean; error?: string } = {};
    try {
      response = JSON.parse(stdout || '{}') as typeof response;
    } catch {
      /* diagnostic below */
    }
    if (result.error || result.status !== 0 || !response.succeeded)
      throw new Error(response.error || stderr || result.error?.message || 'ARC project command failed');
    return response;
  }

  private readRecents(): ArcRecentProject[] {
    try {
      const parsed = JSON.parse(fs.readFileSync(this.recentPath, 'utf8')) as ArcRecentProject[];
      if (!Array.isArray(parsed)) return [];
      return parsed.slice(0, 24).map((entry) => ({ ...entry, missing: !fs.existsSync(entry.descriptorPath) }));
    } catch {
      return [];
    }
  }

  private writeRecents(entries: ArcRecentProject[]): void {
    writeJsonAtomic(this.recentPath, entries.slice(0, 24));
  }

  private touchRecent(project: ArcProjectCandidate): void {
    const current = this.readRecents().filter(
      (entry) => path.resolve(entry.descriptorPath) !== path.resolve(project.descriptorPath),
    );
    current.unshift({
      descriptorPath: project.descriptorPath,
      projectRoot: project.projectRoot,
      guid: project.descriptor.guid,
      name: project.descriptor.name,
      engineVersion: project.descriptor.engineVersion,
      lastOpenedAt: new Date().toISOString(),
      missing: false,
    });
    this.writeRecents(current);
  }
}
