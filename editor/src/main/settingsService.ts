import fs from 'node:fs';
import path from 'node:path';

import type { EditorSettingDescriptor, EditorSettingsSnapshot } from '../common/editorWorkflowTypes';
import type { ArcProjectCandidate } from '../common/projectTypes';

const schema: EditorSettingDescriptor[] = [
  {
    key: 'editor.theme',
    section: 'Editor',
    label: 'Theme',
    description: 'Color theme used by the editor workbench.',
    type: 'enum',
    defaultValue: 'arcDark',
    options: ['arcDark'],
    scopes: ['user'],
  },
  {
    key: 'editor.autosave.enabled',
    section: 'Editor',
    label: 'Autosave',
    description: 'Write recovery generations for dirty authored documents.',
    type: 'boolean',
    defaultValue: true,
    scopes: ['user', 'project'],
  },
  {
    key: 'renderer.qualityTier',
    section: 'Renderer',
    label: 'Quality Tier',
    description: 'Renderer quality profile used by editor viewports.',
    type: 'enum',
    defaultValue: 'auto',
    options: ['auto', 'low', 'standard', 'high'],
    scopes: ['user', 'project'],
  },
  {
    key: 'renderer.renderPath',
    section: 'Renderer',
    label: 'Render Path',
    description: 'Preferred raster path; automatic selection honors adapter capabilities.',
    type: 'enum',
    defaultValue: 'auto',
    options: ['auto', 'forwardPlus', 'deferred'],
    scopes: ['user', 'project'],
    restartRequired: true,
  },
  {
    key: 'renderer.targetFrameMilliseconds',
    section: 'Renderer',
    label: 'Frame Target',
    description: 'Dynamic-resolution frame-time target in milliseconds.',
    type: 'number',
    defaultValue: 16.67,
    minimum: 4,
    maximum: 100,
    step: 0.01,
    scopes: ['user', 'project'],
  },
  {
    key: 'input.translationSnap',
    section: 'Input',
    label: 'Translation Snap',
    description: 'Default translation snapping increment in metres.',
    type: 'number',
    defaultValue: 0.25,
    minimum: 0.001,
    maximum: 1000,
    step: 0.01,
    scopes: ['user', 'project'],
  },
  {
    key: 'input.rotationSnapDegrees',
    section: 'Input',
    label: 'Rotation Snap',
    description: 'Default rotation snapping increment in degrees.',
    type: 'number',
    defaultValue: 15,
    minimum: 0.1,
    maximum: 180,
    step: 0.1,
    scopes: ['user', 'project'],
  },
  {
    key: 'input.scaleSnap',
    section: 'Input',
    label: 'Scale Snap',
    description: 'Default proportional scale snapping increment.',
    type: 'number',
    defaultValue: 0.1,
    minimum: 0.001,
    maximum: 10,
    step: 0.01,
    scopes: ['user', 'project'],
  },
  {
    key: 'cache.localBudgetBytes',
    section: 'Cache',
    label: 'Local Cache Budget',
    description: 'Maximum local derived-data cache size in bytes.',
    type: 'number',
    defaultValue: 50 * 1024 * 1024 * 1024,
    minimum: 1024 * 1024 * 1024,
    maximum: 1024 * 1024 * 1024 * 1024,
    step: 1024 * 1024 * 1024,
    scopes: ['user', 'project'],
  },
  {
    key: 'paths.externalShaderCompiler',
    section: 'Paths & Tools',
    label: 'External Shader Compiler',
    description: 'Optional machine-local compiler executable override.',
    type: 'string',
    defaultValue: '',
    scopes: ['user'],
  },
  {
    key: 'extensions.allowProjectExtensions',
    section: 'Extensions',
    label: 'Allow Project Extensions',
    description: 'Discover compatible project-declared editor extensions.',
    type: 'boolean',
    defaultValue: true,
    scopes: ['user'],
    restartRequired: true,
  },
  {
    key: 'sourceControl.provider',
    section: 'Source Control',
    label: 'Provider',
    description: 'Source-control provider used by the workspace.',
    type: 'enum',
    defaultValue: 'git',
    options: ['git', 'none'],
    scopes: ['user', 'project'],
  },
  {
    key: 'editor.autosave.idleSeconds',
    section: 'Recovery',
    label: 'Idle Delay',
    description: 'Seconds of authoring inactivity before recovery capture is eligible.',
    type: 'number',
    defaultValue: 5,
    minimum: 1,
    maximum: 300,
    step: 1,
    scopes: ['user', 'project'],
  },
  {
    key: 'editor.autosave.minimumIntervalSeconds',
    section: 'Recovery',
    label: 'Minimum Interval',
    description: 'Minimum seconds between recovery generations.',
    type: 'number',
    defaultValue: 120,
    minimum: 10,
    maximum: 3600,
    step: 10,
    scopes: ['user', 'project'],
  },
  {
    key: 'editor.recovery.generations',
    section: 'Recovery',
    label: 'Generations',
    description: 'Maximum recovery generations retained per document.',
    type: 'number',
    defaultValue: 20,
    minimum: 1,
    maximum: 100,
    step: 1,
    scopes: ['user', 'project'],
  },
  {
    key: 'editor.recovery.projectBudgetBytes',
    section: 'Recovery',
    label: 'Project Recovery Budget',
    description: 'Maximum recovery storage per project in bytes.',
    type: 'number',
    defaultValue: 2 * 1024 * 1024 * 1024,
    minimum: 64 * 1024 * 1024,
    maximum: 64 * 1024 * 1024 * 1024,
    step: 64 * 1024 * 1024,
    scopes: ['user', 'project'],
  },
];

const descriptors = new Map(schema.map((descriptor) => [descriptor.key, descriptor]));
const defaults = Object.fromEntries(schema.map((descriptor) => [descriptor.key, descriptor.defaultValue]));

const readObject = (filePath: string): Record<string, unknown> => {
  try {
    const value = JSON.parse(fs.readFileSync(filePath, 'utf8')) as unknown;
    return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
  } catch {
    return {};
  }
};

const writeAtomic = (filePath: string, value: Record<string, unknown>): void => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.tmp-${process.pid}`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  fs.renameSync(temporary, filePath);
};

const validateValue = (descriptor: EditorSettingDescriptor, value: unknown): void => {
  if (descriptor.type === 'boolean' && typeof value !== 'boolean')
    throw new Error(`${descriptor.key} must be a Boolean value`);
  if (descriptor.type === 'string' && typeof value !== 'string') throw new Error(`${descriptor.key} must be a string`);
  if (descriptor.type === 'enum') {
    if (typeof value !== 'string' || !descriptor.options?.includes(value))
      throw new Error(`${descriptor.key} must be one of ${descriptor.options?.join(', ')}`);
  }
  if (descriptor.type === 'number') {
    if (typeof value !== 'number' || !Number.isFinite(value))
      throw new Error(`${descriptor.key} must be a finite number`);
    if (descriptor.minimum !== undefined && value < descriptor.minimum)
      throw new Error(`${descriptor.key} must be at least ${String(descriptor.minimum)}`);
    if (descriptor.maximum !== undefined && value > descriptor.maximum)
      throw new Error(`${descriptor.key} must be at most ${String(descriptor.maximum)}`);
  }
};

export class SettingsService {
  private revision = 1;

  constructor(
    private readonly userSettingsPath: string,
    private readonly activeProject: () => ArcProjectCandidate | null,
  ) {}

  snapshot(): EditorSettingsSnapshot {
    const user = this.validEntries(readObject(this.resolvedUserSettingsPath()));
    const project = this.validEntries(this.readProjectSettings());
    const values = { ...defaults, ...user, ...project };
    const sources: EditorSettingsSnapshot['sources'] = {};
    for (const key of Object.keys(values))
      sources[key] = Object.hasOwn(project, key) ? 'project' : Object.hasOwn(user, key) ? 'user' : 'default';
    return {
      revision: this.revision,
      values,
      sources,
      restartRequired: schema.filter((entry) => entry.restartRequired).map((entry) => entry.key),
      schema,
    };
  }

  update(
    scope: 'user' | 'project',
    changes: Record<string, unknown>,
    expectedRevision: number,
  ): EditorSettingsSnapshot {
    if (expectedRevision !== this.revision) throw new Error('Settings changed; refresh before applying edits');
    const project = this.activeProject();
    if (scope === 'project' && !project?.writable) throw new Error('The active project is not writable');
    for (const [key, value] of Object.entries(changes)) {
      const descriptor = descriptors.get(key);
      if (!descriptor) throw new Error(`Unknown setting '${key}'`);
      if (!descriptor.scopes.includes(scope)) throw new Error(`${key} cannot be stored in ${scope} settings`);
      if (value !== undefined) validateValue(descriptor, value);
    }
    const updates = new Map<string, Record<string, unknown>>();
    for (const [key, value] of Object.entries(changes)) {
      const target = scope === 'user' ? this.resolvedUserSettingsPath() : this.projectSettingsPathForKey(key);
      if (!target) throw new Error('No writable project settings file is available');
      const next = updates.get(target) ?? { ...readObject(target) };
      if (value === undefined) delete next[key];
      else next[key] = value;
      updates.set(target, next);
    }
    for (const [target, values] of updates) writeAtomic(target, values);
    ++this.revision;
    return this.snapshot();
  }

  private validEntries(values: Record<string, unknown>): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(values)) {
      const descriptor = descriptors.get(key);
      if (!descriptor) continue;
      try {
        validateValue(descriptor, value);
        result[key] = value;
      } catch {
        // Invalid persisted values do not shadow safe defaults.
      }
    }
    return result;
  }

  private projectSettingsPathForKey(key: string): string | null {
    const project = this.activeProject();
    if (!project) return null;
    const relative = key.startsWith('renderer.')
      ? project.descriptor.settings.renderer
      : key.startsWith('input.')
        ? project.descriptor.settings.input
        : project.descriptor.settings.editor;
    return path.join(project.projectRoot, relative);
  }

  private resolvedUserSettingsPath(): string {
    const project = this.activeProject();
    return project
      ? path.join(project.projectRoot, project.descriptor.paths.saved, 'Editor', 'settings.v1.json')
      : this.userSettingsPath;
  }

  private readProjectSettings(): Record<string, unknown> {
    const project = this.activeProject();
    if (!project) return {};
    return {
      ...readObject(path.join(project.projectRoot, project.descriptor.settings.editor)),
      ...readObject(path.join(project.projectRoot, project.descriptor.settings.renderer)),
      ...readObject(path.join(project.projectRoot, project.descriptor.settings.input)),
    };
  }
}
