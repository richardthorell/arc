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
export const projectSettingsCatalog = {
  version: 1,
  categories: [
    { id: 'general', label: 'General' },
    { id: 'startup', label: 'Startup' },
    { id: 'targets', label: 'Platforms & Targets' },
    { id: 'build', label: 'Build & Cook' },
  ],
  fields: [
    { id: 'name', category: 'general', label: 'Project name', control: 'text' },
    { id: 'package.applicationName', category: 'general', label: 'Application display name', control: 'text' },
    { id: 'package.companyName', category: 'general', label: 'Company name', control: 'text', defaultValue: '' },
    { id: 'guid', category: 'general', label: 'Project GUID', control: 'readonly' },
    { id: 'engineVersion', category: 'general', label: 'Engine version', control: 'readonly' },
    { id: 'descriptorPath', category: 'general', label: 'Descriptor location', control: 'readonly' },
    { id: 'defaultScene', category: 'startup', label: 'Editor default scene', control: 'scene', defaultValue: null },
    { id: 'startupScenes', category: 'startup', label: 'Runtime startup scenes', control: 'scenes', defaultValue: [] },
    { id: 'targetPlatforms', category: 'targets', label: 'Project targets', control: 'targets' },
    {
      id: 'platformSettings.android.applicationId',
      category: 'targets',
      label: 'Android application ID',
      description: 'For example com.company.game. Required for project APK packaging; not the ARC project GUID.',
      control: 'text',
      defaultValue: '',
    },
    {
      id: 'platformSettings.android.versionName',
      category: 'targets',
      label: 'Android version name',
      control: 'text',
      defaultValue: '1.0',
    },
    {
      id: 'platformSettings.android.versionCode',
      category: 'targets',
      label: 'Android version code',
      control: 'number',
      defaultValue: 1,
    },
    {
      id: 'platformSettings.android.minimumSdk',
      category: 'targets',
      label: 'Android minimum SDK',
      control: 'number',
      defaultValue: 26,
    },
    {
      id: 'platformSettings.android.targetSdk',
      category: 'targets',
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
  catalog: { version: number; categories: { id: string; label: string }[]; fields: ProjectSettingsField[] };
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
