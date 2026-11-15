import { z } from 'zod';

const text = z.string().max(256);
const scene = z
  .object({ guid: z.string().min(1).max(64), expectedType: text, pathHint: z.string().max(2048) })
  .passthrough();
const profile = z
  .object({
    id: text.min(1),
    platform: text.min(1),
    architecture: text.min(1),
    renderer: text.min(1),
    api: text,
    configuration: text.min(1),
    textures: z.object({ outputs: z.array(text).min(1).max(4), quality: text.min(1) }).passthrough(),
  })
  .passthrough();
export const projectSettingSchemas = {
  name: text.min(1),
  'package.applicationName': text.min(1),
  'package.companyName': text,
  'package.output': z.string().min(1).max(2048),
  'package.regionChunks': z.boolean(),
  defaultScene: scene.nullable(),
  startupScenes: z.array(scene).max(64),
  targetPlatforms: z
    .array(z.object({ id: text.min(1), enabled: z.boolean() }).passthrough())
    .min(1)
    .max(32),
  cookProfiles: z.array(profile).max(64),
  'platformSettings.android.applicationId': text.refine(
    (value) => value === '' || /^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+$/u.test(value),
    'Use a dotted application ID, for example com.company.game',
  ),
  'platformSettings.android.versionName': text.min(1).max(128),
  'platformSettings.android.versionCode': z.number().int().min(1).max(2100000000),
  'platformSettings.android.minimumSdk': z.number().int().min(26).max(35),
  'platformSettings.android.targetSdk': z.number().int().min(26).max(35),
};
export type ProjectSettingId = keyof typeof projectSettingSchemas;
export type ProjectSettingsField = {
  id: string;
  category: string;
  label: string;
  description?: string;
  section?: string;
  control: 'text' | 'number' | 'boolean' | 'readonly' | 'scene' | 'scenes' | 'targets' | 'profiles';
  defaultValue?: unknown;
};
export type ProjectSettingsCategory = {
  id: string;
  label: string;
  parentId?: string;
  description?: string;
  planned?: boolean;
  platform?: string;
};
export function visibleProjectSettingsCategories(
  categories: ProjectSettingsCategory[],
  targets: { id: string; enabled: boolean }[],
): ProjectSettingsCategory[] {
  const enabled = new Set(targets.filter((target) => target.enabled).map((target) => target.id.split('-')[0]));
  return categories.filter((category) => !category.planned && (!category.platform || enabled.has(category.platform)));
}
export const projectSettingsCatalog = {
  version: 1,
  categories: [
    { id: 'general-root', label: 'General' },
    { id: 'general', label: 'Overview', parentId: 'general-root' },
    { id: 'startup', label: 'Startup', parentId: 'general-root' },
    { id: 'platforms', label: 'Platforms' },
    { id: 'windows', label: 'Windows', parentId: 'platforms', platform: 'windows' },
    { id: 'android', label: 'Android', parentId: 'platforms', platform: 'android' },
    { id: 'linux', label: 'Linux', parentId: 'platforms', platform: 'linux' },
    { id: 'distribution', label: 'Build & Release' },
    { id: 'build', label: 'Cooking & Packaging', parentId: 'distribution' },
    {
      id: 'rendering',
      label: 'Rendering & Quality',
      description: 'Read-only defaults and the authoritative renderer configuration reference.',
    },
    { id: 'input', label: 'Input', description: 'The input configuration remains authoritative.' },
    {
      id: 'content',
      label: 'Content & Assets',
      description: 'Asset roots are read-only; relocation requires a migration.',
    },
    { id: 'modules', label: 'Plugins & Modules', description: 'Read-only declarations and dependencies.' },
    {
      id: 'advanced',
      label: 'Advanced',
      description:
        'Read-only project paths, compatibility and compiler policy. No workstation settings are stored here.',
    },
  ] satisfies ProjectSettingsCategory[],
  fields: [
    { id: 'name', category: 'general', label: 'Project name', control: 'text' },
    { id: 'package.applicationName', category: 'general', label: 'Application display name', control: 'text' },
    { id: 'package.companyName', category: 'general', label: 'Company name', control: 'text', defaultValue: '' },
    { id: 'guid', category: 'general', label: 'Project GUID', control: 'readonly' },
    { id: 'formatVersion', category: 'advanced', label: 'Project descriptor version', control: 'readonly' },
    { id: 'engineVersion', category: 'general', label: 'Engine version', control: 'readonly' },
    { id: 'descriptorPath', category: 'advanced', label: 'Descriptor location', control: 'readonly' },
    { id: 'defaultScene', category: 'startup', label: 'Editor default scene', control: 'scene', defaultValue: null },
    { id: 'startupScenes', category: 'startup', label: 'Runtime startup scenes', control: 'scenes', defaultValue: [] },
    { id: 'targetPlatforms', category: 'platforms', label: 'Project targets', control: 'targets' },
    {
      id: 'platformSettings.android.applicationId',
      category: 'android',
      section: 'Application',
      label: 'Android application ID',
      description: 'For example com.company.game. Required for project APK packaging; not the ARC project GUID.',
      control: 'text',
      defaultValue: '',
    },
    {
      id: 'platformSettings.android.versionName',
      category: 'android',
      section: 'Application',
      label: 'Android version name',
      control: 'text',
      defaultValue: '1.0',
    },
    {
      id: 'platformSettings.android.versionCode',
      category: 'android',
      section: 'Application',
      label: 'Android version code',
      control: 'number',
      defaultValue: 1,
    },
    {
      id: 'platformSettings.android.minimumSdk',
      category: 'android',
      section: 'Compatibility',
      label: 'Android minimum SDK',
      control: 'number',
      defaultValue: 26,
    },
    {
      id: 'platformSettings.android.targetSdk',
      category: 'android',
      section: 'Compatibility',
      label: 'Android target SDK',
      control: 'number',
      defaultValue: 28,
    },
    { id: 'cookProfiles', category: 'build', label: 'Cook profiles', control: 'profiles', defaultValue: [] },
    {
      id: 'package.output',
      category: 'build',
      label: 'Package output directory',
      description: 'Relative to the project, inside its Build directory.',
      control: 'text',
      defaultValue: 'Build/Packages',
    },
    {
      id: 'package.regionChunks',
      category: 'build',
      label: 'Separate world-region chunks',
      control: 'boolean',
      defaultValue: true,
    },
    ...['source', 'content', 'config', 'plugins', 'build', 'intermediate', 'saved'].map((name) => ({
      id: `paths.${name}`,
      category: 'advanced',
      label: `${name[0].toUpperCase()}${name.slice(1)} root`,
      control: 'readonly' as const,
    })),
    { id: 'assetRoots', category: 'content', label: 'Asset roots', control: 'readonly' },
    { id: 'toolchain', category: 'advanced', label: 'Toolchain policy', control: 'readonly' },
    { id: 'buildConfigurations', category: 'advanced', label: 'Build configurations', control: 'readonly' },
    { id: 'renderer.backend', category: 'rendering', label: 'Renderer backend', control: 'readonly' },
    { id: 'renderer.api', category: 'rendering', label: 'Graphics API requirement', control: 'readonly' },
    { id: 'renderer.quality', category: 'rendering', label: 'Quality default', control: 'readonly' },
    { id: 'settings.renderer', category: 'rendering', label: 'Renderer configuration', control: 'readonly' },
    { id: 'settings.input', category: 'input', label: 'Input mapping configuration', control: 'readonly' },
    { id: 'modules', category: 'modules', label: 'Project modules and dependencies', control: 'readonly' },
    { id: 'plugins', category: 'modules', label: 'Declared plugins', control: 'readonly' },
  ] satisfies ProjectSettingsField[],
};
export const projectSettingsCapabilitiesSchema = z.object({
  version: z.literal(1),
  targets: z.array(z.string()).max(32),
  textureOutputs: z.array(z.string()).max(16),
  textureQualities: z.array(z.string()).max(16),
  androidMinimumSdk: z.number(),
  androidCompileSdk: z.number(),
});
export type ProjectSettingsSnapshot = {
  catalog: { version: number; categories: ProjectSettingsCategory[]; fields: ProjectSettingsField[] };
  projectGuid: string;
  revision: string;
  writable: boolean;
  values: Record<string, unknown>;
  capabilities: z.infer<typeof projectSettingsCapabilitiesSchema>;
  configurations: string[];
  capabilitiesPending?: boolean;
  capabilitiesError?: string;
};
export type ProjectSettingsApplyRequest = {
  catalogVersion: number;
  projectGuid: string;
  revision: string;
  changes: Record<string, unknown>;
};
export type ProjectSettingsApplyResult = {
  succeeded: boolean;
  snapshot?: ProjectSettingsSnapshot;
  errors?: Record<string, string>;
};

export function projectSettingValue(source: unknown, id: string): unknown {
  return id
    .split('.')
    .reduce<unknown>(
      (value, key) => (value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : undefined),
      source,
    );
}

export function validateProjectSettingChanges(changes: Record<string, unknown>): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const [id, value] of Object.entries(changes)) {
    const schema = Object.hasOwn(projectSettingSchemas, id) ? projectSettingSchemas[id as ProjectSettingId] : undefined;
    if (!schema) errors[id] = 'Unknown or read-only project setting';
    else {
      const result = schema.safeParse(value);
      if (!result.success) errors[id] = result.error.issues.map((issue) => issue.message).join('; ');
    }
  }
  return errors;
}
