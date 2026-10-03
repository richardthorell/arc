import fs from 'node:fs';
import path from 'node:path';

import { arcProjectFormat } from '../common/projectTypes';
import type {
  AiInstructionProjectScope,
  AiInstructionSourceDiagnostic,
  AiInstructionSourceSnapshot,
} from '../common/aiInstructionTypes';
import { AiSkillService, type AiSkillProjectScope } from './aiSkillService';

const projectInstructionFileName = 'AGENTS.md';
const maximumProjectInstructionBytes = 256 * 1024;

const normalizedGuid = (value: string) => value.trim().toLocaleLowerCase();

const containedPath = (root: string, candidate: string): boolean => {
  const relative = path.relative(root, candidate);
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
};

const validatedProjectScope = (
  requested: AiInstructionProjectScope | null | undefined,
): { project: AiSkillProjectScope | null; diagnostic?: AiInstructionSourceDiagnostic } => {
  if (!requested) return { project: null };
  try {
    const projectRoot = fs.realpathSync(path.resolve(requested.projectRoot));
    const descriptorPath = fs.realpathSync(path.resolve(requested.descriptorPath));
    if (!containedPath(projectRoot, descriptorPath)) throw new Error('Project descriptor resolves outside the project');
    const descriptor = JSON.parse(fs.readFileSync(descriptorPath, 'utf8')) as {
      format?: unknown;
      guid?: unknown;
    };
    if (descriptor.format !== arcProjectFormat) throw new Error('Project descriptor is not an ARC project');
    if (typeof descriptor.guid !== 'string' || normalizedGuid(descriptor.guid) !== normalizedGuid(requested.projectGuid))
      throw new Error('Project descriptor GUID does not match the active project');
    return { project: { projectRoot, projectGuid: descriptor.guid } };
  } catch (error) {
    return {
      project: null,
      diagnostic: {
        source: 'project-instructions',
        message: `Project instruction scope was rejected: ${error instanceof Error ? error.message : String(error)}`,
      },
    };
  }
};

const readProjectInstructions = (
  project: AiSkillProjectScope | null,
  diagnostics: AiInstructionSourceDiagnostic[],
): string | undefined => {
  if (!project) return undefined;
  const candidate = path.join(project.projectRoot, projectInstructionFileName);
  if (!fs.existsSync(candidate)) return undefined;
  try {
    const projectRoot = fs.realpathSync(project.projectRoot);
    const instructionPath = fs.realpathSync(candidate);
    if (!containedPath(projectRoot, instructionPath)) throw new Error(`${projectInstructionFileName} resolves outside the project`);
    const stat = fs.statSync(instructionPath);
    if (!stat.isFile()) throw new Error(`${projectInstructionFileName} must be a file`);
    if (stat.size > maximumProjectInstructionBytes)
      throw new Error(`${projectInstructionFileName} exceeds the ${maximumProjectInstructionBytes} byte limit`);
    const instructions = fs.readFileSync(instructionPath, 'utf8').trim();
    return instructions || undefined;
  } catch (error) {
    diagnostics.push({
      source: 'project-instructions',
      message: error instanceof Error ? error.message : String(error),
    });
    return undefined;
  }
};

export const loadAiInstructionSources = (
  builtinRoot: string,
  requestedProject?: AiInstructionProjectScope | null,
): AiInstructionSourceSnapshot => {
  const validation = validatedProjectScope(requestedProject);
  const diagnostics: AiInstructionSourceDiagnostic[] = [];
  if (validation.diagnostic) diagnostics.push(validation.diagnostic);

  const skillSnapshot = new AiSkillService({
    builtinRoot,
    project: () => validation.project,
  }).snapshot(true);
  for (const diagnostic of skillSnapshot.diagnostics) {
    diagnostics.push({
      source: diagnostic.origin === 'builtin' ? 'builtin-skills' : 'project-skills',
      message: diagnostic.message,
    });
  }

  const projectInstructions = readProjectInstructions(validation.project, diagnostics);
  return {
    revision: skillSnapshot.revision,
    projectGuid: validation.project?.projectGuid ?? null,
    ...(projectInstructions ? { projectInstructions } : {}),
    skills: skillSnapshot.skills.map((skill) => ({
      manifest: skill.manifest,
      instructions: skill.instructions,
      origin: skill.origin,
      projectGuid: skill.projectGuid,
    })),
    diagnostics,
  };
};
