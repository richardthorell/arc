import type { AiContextSectionId } from './aiContextTypes';
import type { AiSkillCapability, AiSkillManifest, AiSkillOrigin } from './aiSkillTypes';

export type AiInstructionProjectScope = {
  projectRoot: string;
  descriptorPath: string;
  projectGuid: string;
};

export type AiInstructionSkill = {
  manifest: AiSkillManifest;
  instructions: string;
  origin: AiSkillOrigin;
  projectGuid: string | null;
};

export type AiInstructionSourceDiagnostic = {
  source: 'builtin-skills' | 'project-skills' | 'project-instructions';
  message: string;
};

export type AiInstructionSourceSnapshot = {
  revision: number;
  projectGuid: string | null;
  projectInstructions?: string;
  skills: AiInstructionSkill[];
  diagnostics: AiInstructionSourceDiagnostic[];
};

export type AiInstructionSourceRequest = {
  project?: AiInstructionProjectScope | null;
};

export type AiSkillResolutionReason = 'selected' | 'not-relevant' | 'missing-capability';

export type AiResolvedSkillDiagnostic = {
  id: string;
  origin: AiSkillOrigin;
  score: number;
  reason: AiSkillResolutionReason;
  missingCapabilities: AiSkillCapability[];
  declaredTools: string[];
  availableDeclaredTools: string[];
  contexts: AiContextSectionId[];
};

export type AiInstructionResolutionDiagnostics = {
  instructionSources: Array<{
    kind: 'base' | 'project' | 'skill';
    id: string;
  }>;
  selectedSkillIds: string[];
  availableCapabilities: AiSkillCapability[];
  availableTools: string[];
  availableContexts: AiContextSectionId[];
  skills: AiResolvedSkillDiagnostic[];
  sourceDiagnostics: AiInstructionSourceDiagnostic[];
};
