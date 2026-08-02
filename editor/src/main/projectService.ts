import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
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

const parseDescriptor = (value: unknown): ArcProjectDescriptor => {
  if (!value || typeof value !== 'object') throw new Error('Project descriptor must be a JSON object');
  const source = value as Partial<ArcProjectDescriptor>;
  if (source.format !== arcProjectFormat) throw new Error(`Expected project format '${arcProjectFormat}'`);
  if (source.formatVersion !== arcProjectFormatVersion)
    throw new Error(`Unsupported project format version ${String(source.formatVersion)}`);
  if (
    typeof source.guid !== 'string' ||
    !/^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i.test(source.guid)
  )
    throw new Error('Project GUID is missing or malformed');
  if (typeof source.name !== 'string' || !source.name.trim()) throw new Error('Project name is required');
  if (typeof source.engineVersion !== 'string' || !source.engineVersion.trim())
    throw new Error('Project engine version is required');
  return {
    format: arcProjectFormat,
    formatVersion: arcProjectFormatVersion,
    guid: source.guid,
    name: source.name.trim(),
    engineVersion: source.engineVersion,
    assetRoots: normalizeStringArray(source.assetRoots).length
      ? normalizeProjectRelativePaths(source.assetRoots, 'assetRoots')
      : ['assets'],
    startupScenes: normalizeProjectRelativePaths(source.startupScenes, 'startupScenes'),
    modules: normalizeStringArray(source.modules),
    extensions: normalizeProjectRelativePaths(source.extensions, 'extensions'),
    settings: {
      editor: normalizeProjectRelativePath(source.settings?.editor || 'config/editor.settings.json', 'settings.editor'),
      renderer: normalizeProjectRelativePath(
        source.settings?.renderer || 'config/renderer.settings.json',
        'settings.renderer',
      ),
      input: normalizeProjectRelativePath(source.settings?.input || 'config/input.settings.json', 'settings.input'),
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
  private readonly host: ProjectHost;
  private activeProject: ArcProjectCandidate | null = null;

  constructor(options: {
    userDataPath: string;
    currentEngineVersion: string;
    currentEditorPath: string;
    host: ProjectHost;
  }) {
    this.recentPath = path.join(options.userDataPath, recentFileName);
    this.currentEngineVersion = options.currentEngineVersion;
    this.currentEditorPath = options.currentEditorPath;
    this.host = options.host;
  }

  snapshot(): ArcProjectBrowserSnapshot {
    return {
      currentEngineVersion: this.currentEngineVersion,
      activeProject: this.activeProject,
      recentProjects: this.readRecents(),
      installations: this.engineInstallations(),
      hostConnected: this.host.connected,
      hostError: this.host.error,
    };
  }

  active(): ArcProjectCandidate | null {
    return this.activeProject;
  }

  inspect(candidate: string): ArcProjectCandidate {
    const descriptorPath = resolveDescriptorPath(candidate);
    const descriptor = parseDescriptor(JSON.parse(fs.readFileSync(descriptorPath, 'utf8')) as unknown);
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
      const response = await this.host.command('project.open', {
        name: project.descriptor.name,
        root: project.projectRoot,
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

  create(request: ArcCreateProjectRequest): ArcProjectOperationResult {
    try {
      const destination = path.resolve(request.destination);
      if (fs.existsSync(destination) && fs.readdirSync(destination).length)
        throw new Error('Project destination must be empty');
      fs.mkdirSync(path.join(destination, 'assets', 'scenes'), { recursive: true });
      fs.mkdirSync(path.join(destination, 'config'), { recursive: true });
      const safeName = request.name.trim();
      if (!safeName) throw new Error('Project name is required');
      const descriptor: ArcProjectDescriptor = {
        format: arcProjectFormat,
        formatVersion: arcProjectFormatVersion,
        guid: crypto.randomUUID(),
        name: safeName,
        engineVersion: this.currentEngineVersion,
        assetRoots: ['assets'],
        startupScenes: [],
        modules: ['arc-framework', 'arc-render', 'arc-scene'],
        extensions: [],
        settings: {
          editor: 'config/editor.settings.json',
          renderer: 'config/renderer.settings.json',
          input: 'config/input.settings.json',
        },
      };
      const descriptorPath = path.join(destination, `${safeName.replace(/[^a-z0-9_-]+/gi, '-')}${descriptorExtension}`);
      writeJsonAtomic(descriptorPath, descriptor);
      writeJsonAtomic(path.join(destination, descriptor.settings.editor), { formatVersion: 1 });
      writeJsonAtomic(path.join(destination, descriptor.settings.renderer), { formatVersion: 1 });
      writeJsonAtomic(path.join(destination, descriptor.settings.input), { formatVersion: 1 });
      return { succeeded: true, project: this.inspect(descriptorPath) };
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

  private upgrade(project: ArcProjectCandidate): ArcProjectCandidate {
    if (project.compatibility !== 'upgradeRequired') return project;
    const backup = `${project.descriptorPath}.pre-${this.currentEngineVersion}.bak`;
    fs.copyFileSync(project.descriptorPath, backup);
    parseDescriptor(JSON.parse(fs.readFileSync(backup, 'utf8')) as unknown);
    try {
      writeJsonAtomic(project.descriptorPath, {
        ...project.descriptor,
        engineVersion: this.currentEngineVersion,
      });
      const upgraded = this.inspect(project.descriptorPath);
      if (upgraded.compatibility !== 'compatible')
        throw new Error('Project upgrade did not produce a compatible descriptor');
      return upgraded;
    } catch (error) {
      fs.copyFileSync(backup, project.descriptorPath);
      throw error;
    }
  }

  private engineInstallations(): ArcEngineInstallation[] {
    const installations: ArcEngineInstallation[] = [
      { version: this.currentEngineVersion, editorPath: this.currentEditorPath, current: true },
    ];
    const configured = process.env.ARC_ENGINE_INSTALLS?.split(path.delimiter).filter(Boolean) ?? [];
    for (const editorPath of configured) {
      if (
        !fs.existsSync(editorPath) ||
        installations.some((entry) => path.resolve(entry.editorPath) === path.resolve(editorPath))
      )
        continue;
      installations.push({ version: path.basename(path.dirname(editorPath)), editorPath, current: false });
    }
    return installations;
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
