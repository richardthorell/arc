import fs from 'node:fs';
import path from 'node:path';

import type { ArcExtensionCapability, ArcExtensionManifest, ArcExtensionSnapshot } from '../common/extensionTypes';
import type { ArcProjectCandidate } from '../common/projectTypes';

const validCapabilities = new Set<ArcExtensionCapability>([
  'filesystem.read',
  'filesystem.write',
  'sourceControl',
  'asset.read',
  'asset.mutate',
  'scene.read',
  'scene.mutate',
]);

const stringArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === 'string') : [];

export class ExtensionService {
  private readonly project: () => ArcProjectCandidate | null;
  private readonly engineVersion: string;
  private readonly enabled: () => boolean;
  private revision = 0;
  private current: ArcExtensionSnapshot = { revision: 0, extensions: [] };

  constructor(project: () => ArcProjectCandidate | null, engineVersion: string, enabled: () => boolean = () => true) {
    this.project = project;
    this.engineVersion = engineVersion;
    this.enabled = enabled;
  }

  snapshot(force = false): ArcExtensionSnapshot {
    if (!force && this.current.revision) return this.current;
    const project = this.project();
    const extensions: ArcExtensionSnapshot['extensions'] = [];
    const extensionIds = new Set<string>();
    if (project && this.enabled()) {
      for (const configured of project.descriptor.plugins
        .filter((plugin) => plugin.enabled && plugin.path)
        .map((plugin) => plugin.path as string)) {
        let root = path.resolve(project.projectRoot, configured);
        const diagnostics: string[] = [];
        try {
          root = this.resolveInsideProject(project.projectRoot, configured);
          const manifestPath = path.join(root, 'arc-extension.json');
          const manifest = this.parse(JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as unknown, root);
          if (extensionIds.has(manifest.id)) throw new Error(`Duplicate extension ID '${manifest.id}'`);
          extensionIds.add(manifest.id);
          const compatible = manifest.engineVersion === this.engineVersion;
          if (!compatible) diagnostics.push(`Requires ARC ${manifest.engineVersion}; running ${this.engineVersion}`);
          extensions.push({
            manifest,
            root,
            compatible,
            enabled: compatible,
            grantedCapabilities: manifest.capabilities.filter(
              (capability) => capability === 'asset.read' || capability === 'scene.read',
            ),
            diagnostics,
          });
        } catch (error) {
          diagnostics.push(error instanceof Error ? error.message : String(error));
          extensions.push({
            manifest: {
              format: 'arc-extension',
              formatVersion: 1,
              id: configured,
              name: path.basename(configured),
              version: 'invalid',
              engineVersion: '',
              main: '',
              activationEvents: [],
              capabilities: [],
            },
            root,
            compatible: false,
            enabled: false,
            grantedCapabilities: [],
            diagnostics,
          });
        }
      }
    }
    this.current = { revision: ++this.revision, extensions };
    return this.current;
  }

  invalidate(): void {
    this.current = { revision: 0, extensions: [] };
  }

  private parse(value: unknown, root: string): ArcExtensionManifest {
    if (!value || typeof value !== 'object') throw new Error('Extension manifest must be an object');
    const source = value as Partial<ArcExtensionManifest>;
    if (source.format !== 'arc-extension' || source.formatVersion !== 1)
      throw new Error('Unsupported extension manifest format');
    if (typeof source.id !== 'string' || !/^[a-z0-9][a-z0-9._-]+$/i.test(source.id))
      throw new Error('Extension ID is malformed');
    if (typeof source.name !== 'string' || !source.name.trim()) throw new Error('Extension name is required');
    if (typeof source.version !== 'string' || !/^\d+\.\d+\.\d+(?:[-+][0-9a-z.-]+)?$/i.test(source.version))
      throw new Error('Extension version must be semantic');
    if (typeof source.engineVersion !== 'string' || !source.engineVersion.trim())
      throw new Error('Extension identity fields are required');
    if (typeof source.main !== 'string') throw new Error('Extension entry point must remain inside the extension');
    const mainEntry = source.main;
    const main = this.resolveExtensionEntry(root, mainEntry, 'Extension entry point');
    if (!fs.statSync(main).isFile()) throw new Error('Extension entry point must be a file');
    const capabilities = stringArray(source.capabilities);
    if (capabilities.some((value) => !validCapabilities.has(value as ArcExtensionCapability)))
      throw new Error('Extension requests an unknown capability');
    const contributes = this.validateContributions(source.contributes, root);
    return {
      format: 'arc-extension',
      formatVersion: 1,
      id: source.id,
      name: source.name.trim(),
      version: source.version,
      engineVersion: source.engineVersion,
      main: mainEntry,
      activationEvents: stringArray(source.activationEvents),
      capabilities: capabilities as ArcExtensionCapability[],
      contributes,
    };
  }

  private validateContributions(
    value: ArcExtensionManifest['contributes'] | undefined,
    root: string,
  ): ArcExtensionManifest['contributes'] | undefined {
    if (value === undefined) return undefined;
    if (!value || typeof value !== 'object') throw new Error('Extension contributions must be an object');
    const identifier = /^[a-z0-9][a-z0-9._-]+$/i;
    const commands = value.commands?.map((command) => {
      if (!command || !identifier.test(command.id) || !command.title.trim())
        throw new Error('Extension command contribution is malformed');
      return { id: command.id, title: command.title.trim() };
    });
    const panels = value.panels?.map((panel) => {
      if (!panel || !identifier.test(panel.id) || !panel.title.trim())
        throw new Error('Extension panel contribution is malformed');
      this.resolveExtensionEntry(root, panel.entry, `Panel '${panel.id}' entry point`);
      return { id: panel.id, title: panel.title.trim(), entry: panel.entry };
    });
    const propertyDrawers = value.propertyDrawers?.map((drawer) => {
      if (!drawer || !drawer.fieldType.trim()) throw new Error('Extension property drawer contribution is malformed');
      this.resolveExtensionEntry(root, drawer.entry, `Property drawer '${drawer.fieldType}' entry point`);
      return drawer;
    });
    const assetEditors = value.assetEditors?.map((editor) => {
      if (!editor || !editor.assetType.trim()) throw new Error('Extension asset editor contribution is malformed');
      this.resolveExtensionEntry(root, editor.entry, `Asset editor '${editor.assetType}' entry point`);
      return editor;
    });
    return { commands, panels, propertyDrawers, assetEditors };
  }

  private resolveExtensionEntry(root: string, entry: unknown, label: string): string {
    if (typeof entry !== 'string' || !entry.trim() || path.isAbsolute(entry) || entry.split(/[\\/]/).includes('..'))
      throw new Error(`${label} must remain inside the extension`);
    const resolved = path.resolve(root, entry);
    const relative = path.relative(root, resolved);
    if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative))
      throw new Error(`${label} must remain inside the extension`);
    if (!fs.existsSync(resolved)) throw new Error(`${label} does not exist`);
    const realRoot = fs.realpathSync(root);
    const realEntry = fs.realpathSync(resolved);
    const realRelative = path.relative(realRoot, realEntry);
    if (
      !realRelative ||
      realRelative === '..' ||
      realRelative.startsWith(`..${path.sep}`) ||
      path.isAbsolute(realRelative)
    )
      throw new Error(`${label} resolves outside the extension`);
    return resolved;
  }

  private resolveInsideProject(projectRoot: string, relative: string): string {
    if (path.isAbsolute(relative)) throw new Error('Extension paths must be project-relative');
    const resolved = path.resolve(projectRoot, relative);
    const back = path.relative(projectRoot, resolved);
    if (!back || back === '..' || back.startsWith(`..${path.sep}`) || path.isAbsolute(back))
      throw new Error('Extension path escapes the project');
    if (!fs.existsSync(resolved) || !fs.statSync(resolved).isDirectory())
      throw new Error(`Extension directory does not exist: ${relative}`);
    const realProject = fs.realpathSync(projectRoot);
    const realExtension = fs.realpathSync(resolved);
    const realBack = path.relative(realProject, realExtension);
    if (!realBack || realBack === '..' || realBack.startsWith(`..${path.sep}`) || path.isAbsolute(realBack))
      throw new Error('Extension path resolves outside the project');
    return resolved;
  }
}
