import fs from 'node:fs';
import { createRequire } from 'node:module';
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
    optionLabels: { arcDark: 'Dark (Default)' },
    scopes: ['user'],
  },
  {
    key: 'ai.openai.apiKey',
    section: 'AI Providers',
    label: 'API Key',
    description: 'OpenAI project API key. Stored encrypted on this machine and never written to project settings.',
    type: 'string',
    format: 'secret',
    defaultValue: '',
    scopes: ['user'],
  },
  {
    key: 'ai.openai.model',
    section: 'AI Providers',
    label: 'Model',
    description: 'Default OpenAI model used by ARC Assistant conversations.',
    type: 'enum',
    defaultValue: 'gpt-6-sol',
    options: ['gpt-6-astra', 'gpt-6-sol', 'gpt-6-luna'],
    optionLabels: {
      'gpt-6-astra': 'GPT-6 Astra',
      'gpt-6-sol': 'GPT-6 Sol',
      'gpt-6-luna': 'GPT-6 Luna',
    },
    scopes: ['user'],
  },
  {
    key: 'ai.openai.reasoningEffort',
    section: 'AI Providers',
    label: 'Reasoning Effort',
    description: 'Default reasoning effort for OpenAI model responses.',
    type: 'enum',
    defaultValue: 'medium',
    options: ['low', 'medium', 'high', 'xhigh', 'max'],
    optionLabels: { low: 'Low', medium: 'Medium', high: 'High', xhigh: 'Extra High', max: 'Maximum' },
    scopes: ['user'],
  },
  {
    key: 'ai.openai.organizationId',
    section: 'AI Providers',
    label: 'Organization ID',
    description: 'Optional OpenAI organization override for accounts that belong to multiple organizations.',
    type: 'string',
    defaultValue: '',
    scopes: ['user'],
  },
  {
    key: 'ai.openai.projectId',
    section: 'AI Providers',
    label: 'Project ID',
    description: 'Optional OpenAI project override. Project API keys normally select this automatically.',
    type: 'string',
    defaultValue: '',
    scopes: ['user'],
  },
  {
    key: 'ai.openai.storeResponses',
    section: 'AI Providers',
    label: 'Store Responses',
    description: 'Allow OpenAI to retain Responses API objects. ARC keeps this disabled by default.',
    type: 'boolean',
    defaultValue: false,
    scopes: ['user'],
  },
  {
    key: 'editor.autosave.enabled',
    section: 'Recovery',
    label: 'Autosave',
    description: 'Write recovery generations for dirty authored documents.',
    type: 'boolean',
    defaultValue: true,
    scopes: ['user', 'project'],
  },
  {
    key: 'renderer.gridColor',
    section: 'Renderer',
    label: 'Grid Color',
    description: 'Base color used by the editor viewport grid.',
    type: 'string',
    format: 'color',
    defaultValue: '#33373D',
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
    key: 'renderer.antiAliasing',
    section: 'Renderer',
    label: 'Anti-Aliasing',
    description: 'Default anti-aliasing policy inherited by cameras and viewports.',
    type: 'enum',
    defaultValue: 'auto',
    options: ['auto', 'disabled', 'fxaa', 'taa', 'taau'],
    scopes: ['user', 'project'],
  },
  {
    key: 'renderer.temporal.historyWeight',
    section: 'Renderer',
    label: 'Temporal History Weight',
    description: 'Contribution retained from validated temporal history samples.',
    type: 'number',
    defaultValue: 0.9,
    minimum: 0,
    maximum: 0.98,
    step: 0.01,
    scopes: ['user', 'project'],
  },
  {
    key: 'renderer.temporal.disocclusionThreshold',
    section: 'Renderer',
    label: 'Disocclusion Threshold',
    description: 'Device-depth difference that rejects a reprojected history sample.',
    type: 'number',
    defaultValue: 0.01,
    minimum: 0.0001,
    maximum: 0.25,
    step: 0.001,
    scopes: ['user', 'project'],
  },
  {
    key: 'renderer.temporal.reactiveResponse',
    section: 'Renderer',
    label: 'Reactive Response',
    description: 'History rejection response for transparency and rapidly changing radiance.',
    type: 'number',
    defaultValue: 1,
    minimum: 0,
    maximum: 4,
    step: 0.05,
    scopes: ['user', 'project'],
  },
  {
    key: 'renderer.temporal.sharpening',
    section: 'Renderer',
    label: 'Temporal Sharpening',
    description: 'Spatial sharpening applied after TAA or temporal upscaling.',
    type: 'number',
    defaultValue: 0.2,
    minimum: 0,
    maximum: 1,
    step: 0.01,
    scopes: ['user', 'project'],
  },
  {
    key: 'renderer.temporal.jitterSamples',
    section: 'Renderer',
    label: 'Jitter Sequence Length',
    description: 'Number of Halton samples used before the camera jitter sequence repeats.',
    type: 'number',
    defaultValue: 8,
    minimum: 1,
    maximum: 32,
    step: 1,
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
    key: 'platform.windows.visualStudioPath',
    section: 'Windows',
    label: 'Visual Studio Installation',
    description: 'Optional Visual Studio installation root. Leave empty to auto-detect Visual Studio.',
    type: 'string',
    defaultValue: '',
    scopes: ['user'],
  },
  {
    key: 'platform.windows.msvcToolchainPath',
    section: 'Windows',
    label: 'MSVC Toolchain',
    description:
      'Optional MSVC toolchain root (VC/Tools/MSVC/<version>). Leave empty to use the detected installation.',
    type: 'string',
    defaultValue: '',
    scopes: ['user'],
  },
  {
    key: 'platform.windows.sdkPath',
    section: 'Windows',
    label: 'Windows SDK',
    description: 'Optional Windows SDK root. Leave empty to use the SDK discovered by the native toolchain.',
    type: 'string',
    defaultValue: '',
    scopes: ['user'],
  },
  {
    key: 'platform.windows.cmakePath',
    section: 'Windows',
    label: 'CMake',
    description: 'Optional path to cmake.exe. Leave empty to use CMake from PATH or Visual Studio.',
    type: 'string',
    defaultValue: '',
    scopes: ['user'],
  },
  {
    key: 'platform.windows.ninjaPath',
    section: 'Windows',
    label: 'Ninja',
    description: 'Optional path to ninja.exe. Leave empty to use Ninja from PATH or Visual Studio.',
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
const secretSettingKeys = new Set(['ai.openai.apiKey']);

export type SettingsSecretCodec = {
  encrypt(value: string): string;
  decrypt(value: string): string;
};

type ElectronSafeStorage = {
  isEncryptionAvailable(): boolean;
  encryptString(value: string): Buffer;
  decryptString(value: Buffer): string;
};

const electronSecretCodec: SettingsSecretCodec = {
  encrypt(value) {
    const electron = createRequire(import.meta.url)('electron') as { safeStorage?: ElectronSafeStorage };
    const storage = electron.safeStorage;
    if (!storage?.isEncryptionAvailable()) throw new Error('Secure credential storage is unavailable on this machine');
    return storage.encryptString(value).toString('base64');
  },
  decrypt(value) {
    const electron = createRequire(import.meta.url)('electron') as { safeStorage?: ElectronSafeStorage };
    const storage = electron.safeStorage;
    if (!storage?.isEncryptionAvailable()) throw new Error('Secure credential storage is unavailable on this machine');
    return storage.decryptString(Buffer.from(value, 'base64'));
  },
};

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
  if (descriptor.format === 'color' && (typeof value !== 'string' || !/^#[0-9a-fA-F]{6}$/.test(value)))
    throw new Error(`${descriptor.key} must be a #RRGGBB color`);
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
    private readonly secretCodec: SettingsSecretCodec = electronSecretCodec,
  ) {}

  snapshot(): EditorSettingsSnapshot {
    const user = this.validEntries(readObject(this.resolvedUserSettingsPath()));
    const project = this.validEntries(this.readProjectSettings());
    const values = { ...defaults, ...user, ...project };
    const sources: EditorSettingsSnapshot['sources'] = {};
    for (const key of Object.keys(values))
      sources[key] = Object.hasOwn(project, key) ? 'project' : Object.hasOwn(user, key) ? 'user' : 'default';

    const secrets = readObject(this.secretSettingsPath());
    for (const key of secretSettingKeys) {
      if (typeof secrets[key] !== 'string') continue;
      values[key] = 'configured';
      sources[key] = 'user';
    }

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
    let secrets: Record<string, unknown> | null = null;
    for (const [key, value] of Object.entries(changes)) {
      if (secretSettingKeys.has(key)) {
        if (scope !== 'user') throw new Error(`${key} cannot be stored in ${scope} settings`);
        secrets ??= { ...readObject(this.secretSettingsPath()) };
        if (value === undefined || value === '') delete secrets[key];
        else secrets[key] = this.secretCodec.encrypt(String(value).trim());
        continue;
      }

      const target = scope === 'user' ? this.resolvedUserSettingsPath() : this.projectSettingsPathForKey(key);
      if (!target) throw new Error('No writable project settings file is available');
      const next = updates.get(target) ?? { ...readObject(target) };
      if (value === undefined) delete next[key];
      else next[key] = value;
      updates.set(target, next);
    }
    for (const [target, values] of updates) writeAtomic(target, values);
    if (secrets) writeAtomic(this.secretSettingsPath(), secrets);
    ++this.revision;
    return this.snapshot();
  }

  secretValue(key: string): string | null {
    if (!secretSettingKeys.has(key)) throw new Error(`'${key}' is not a secret setting`);
    const encrypted = readObject(this.secretSettingsPath())[key];
    if (typeof encrypted !== 'string') return null;
    return this.secretCodec.decrypt(encrypted);
  }

  private validEntries(values: Record<string, unknown>): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(values)) {
      if (secretSettingKeys.has(key)) continue;
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

  private secretSettingsPath(): string {
    return `${this.userSettingsPath}.secrets`;
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
