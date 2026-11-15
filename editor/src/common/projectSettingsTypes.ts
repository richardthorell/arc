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
  control: 'text' | 'number' | 'boolean' | 'readonly' | 'scene' | 'scenes' | 'targets' | 'profiles';
  defaultValue?: unknown;
};
export type ProjectSettingsCategory = {
  id: string;
  label: string;
  parentId?: string;
  description?: string;
  planned?: boolean;
};
export const projectSettingsCatalog = {
  version: 1,
  categories: [
    { id: 'general-root', label: 'General' },
    { id: 'general', label: 'Identity & Metadata', parentId: 'general-root' },
    { id: 'startup', label: 'Startup & Scenes', parentId: 'general-root' },
    {
      id: 'paths',
      label: 'Project Paths',
      parentId: 'general-root',
      description: 'Read-only. Relocating established roots requires a separate migration.',
    },
    { id: 'platforms', label: 'Platforms' },
    { id: 'targets', label: 'Supported Platforms', parentId: 'platforms' },
    {
      id: 'windows',
      label: 'Windows',
      parentId: 'platforms',
      planned: true,
      description:
        'Window and distribution overrides need a validated project contract and finalizer integration. Existing Windows build workflows remain available.',
    },
    { id: 'android', label: 'Android', parentId: 'platforms' },
    { id: 'android-application', label: 'Application', parentId: 'android' },
    {
      id: 'android-compatibility',
      label: 'Compatibility & Display',
      parentId: 'android',
      description:
        'SDK identity is supported. Orientation, screen behavior and cutout policies are planned; compile SDK and NDK are owned by the toolchain.',
    },
    {
      id: 'android-publishing',
      label: 'Publishing & Signing',
      parentId: 'android',
      planned: true,
      description:
        'Release signing and publishing integration is planned. Never store private keys, passwords or tokens in project settings.',
    },
    {
      id: 'other-platforms',
      label: 'Other Platforms',
      parentId: 'platforms',
      planned: true,
      description:
        'Apple, Web and console application settings require their platform backends and finalizers. Available headless targets are listed under Supported Platforms.',
    },
    { id: 'distribution', label: 'Build & Distribution' },
    {
      id: 'build-profiles',
      label: 'Build Profiles',
      parentId: 'distribution',
      planned: true,
      description:
        'Named build profiles and inheritance are separate from existing cook profiles. This page will use the shared workflow contract, not a second configuration system.',
    },
    { id: 'compile', label: 'Compile & Diagnostics', parentId: 'distribution' },
    { id: 'build', label: 'Cooking', parentId: 'distribution' },
    { id: 'packaging', label: 'Packaging', parentId: 'distribution' },
    {
      id: 'rendering',
      label: 'Rendering & Quality',
      description:
        'Read-only descriptor defaults. Detailed quality and device-profile authoring belongs to the existing renderer configuration; editing it here is not yet integrated.',
    },
    {
      id: 'input',
      label: 'Input',
      description: 'The input configuration remains authoritative. Detailed mapping editor integration is planned.',
    },
    {
      id: 'physics',
      label: 'Physics',
      planned: true,
      description:
        'Project-level physics authoring is planned and requires subsystem validation and runtime integration.',
    },
    {
      id: 'audio',
      label: 'Audio',
      planned: true,
      description:
        'Project-level audio authoring is planned and requires subsystem validation and runtime integration.',
    },
    {
      id: 'runtime',
      label: 'Runtime',
      planned: true,
      description:
        'Lifecycle, timing, diagnostics, content delivery and dedicated-server policies need shared runtime contracts before they become editable here.',
    },
    {
      id: 'content',
      label: 'Content & Assets',
      description:
        'Content roots are read-only. Import policy and content delivery settings require their existing authoring services.',
    },
    {
      id: 'modules',
      label: 'Plugins & Modules',
      description: 'Read-only declarations. Dependency-aware module and plugin editing is planned.',
    },
  ] satisfies ProjectSettingsCategory[],
  fields: [
    { id: 'name', category: 'general', label: 'Project name', control: 'text' },
    { id: 'package.applicationName', category: 'general', label: 'Application display name', control: 'text' },
    { id: 'package.companyName', category: 'general', label: 'Company name', control: 'text', defaultValue: '' },
    { id: 'guid', category: 'general', label: 'Project GUID', control: 'readonly' },
    { id: 'formatVersion', category: 'general', label: 'Project descriptor version', control: 'readonly' },
    { id: 'engineVersion', category: 'general', label: 'Engine version', control: 'readonly' },
    { id: 'descriptorPath', category: 'general', label: 'Descriptor location', control: 'readonly' },
    { id: 'defaultScene', category: 'startup', label: 'Editor default scene', control: 'scene', defaultValue: null },
    { id: 'startupScenes', category: 'startup', label: 'Runtime startup scenes', control: 'scenes', defaultValue: [] },
    { id: 'targetPlatforms', category: 'targets', label: 'Project targets', control: 'targets' },
    {
      id: 'platformSettings.android.applicationId',
      category: 'android-application',
      label: 'Android application ID',
      description: 'For example com.company.game. Required for project APK packaging; not the ARC project GUID.',
      control: 'text',
      defaultValue: '',
    },
    {
      id: 'platformSettings.android.versionName',
      category: 'android-application',
      label: 'Android version name',
      control: 'text',
      defaultValue: '1.0',
    },
    {
      id: 'platformSettings.android.versionCode',
      category: 'android-application',
      label: 'Android version code',
      control: 'number',
      defaultValue: 1,
    },
    {
      id: 'platformSettings.android.minimumSdk',
      category: 'android-compatibility',
      label: 'Android minimum SDK',
      control: 'number',
      defaultValue: 26,
    },
    {
      id: 'platformSettings.android.targetSdk',
      category: 'android-compatibility',
      label: 'Android target SDK',
      control: 'number',
      defaultValue: 28,
    },
    { id: 'cookProfiles', category: 'build', label: 'Cook profiles', control: 'profiles', defaultValue: [] },
    {
      id: 'package.output',
      category: 'packaging',
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
      category: 'paths',
      label: `${name[0].toUpperCase()}${name.slice(1)} root`,
      control: 'readonly' as const,
    })),
    { id: 'assetRoots', category: 'content', label: 'Asset roots', control: 'readonly' },
    { id: 'toolchain', category: 'compile', label: 'Toolchain policy', control: 'readonly' },
    { id: 'buildConfigurations', category: 'compile', label: 'Build configurations', control: 'readonly' },
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
