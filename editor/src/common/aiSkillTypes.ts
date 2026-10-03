import type { AiContextSectionId } from './aiContextTypes';

export const AI_SKILL_FORMAT = 'arc-skill' as const;
export const AI_SKILL_FORMAT_VERSION = 1 as const;

export const aiSkillCapabilities = [
  'scene.read',
  'scene.mutate',
  'asset.read',
  'asset.mutate',
  'viewport.read',
  'viewport.control',
  'diagnostics.read',
  'play.control',
] as const;

export type AiSkillCapability = (typeof aiSkillCapabilities)[number];
export type AiSkillOrigin = 'builtin' | 'project';

export type AiSkillManifest = {
  format: typeof AI_SKILL_FORMAT;
  formatVersion: typeof AI_SKILL_FORMAT_VERSION;
  id: string;
  name: string;
  version: string;
  description: string;
  requiredCapabilities: AiSkillCapability[];
  tools: string[];
  contexts: AiContextSectionId[];
};

export type AiSkill = {
  manifest: AiSkillManifest;
  instructions: string;
  origin: AiSkillOrigin;
  root: string;
  filePath: string;
  projectGuid: string | null;
};

export type AiSkillDiagnostic = {
  origin: AiSkillOrigin;
  path: string;
  message: string;
};

export type AiSkillSnapshot = {
  revision: number;
  projectGuid: string | null;
  skills: AiSkill[];
  diagnostics: AiSkillDiagnostic[];
};
