import fs from 'node:fs';
import path from 'node:path';

import {
  AI_SKILL_FORMAT,
  AI_SKILL_FORMAT_VERSION,
  aiSkillCapabilities,
  type AiSkill,
  type AiSkillCapability,
  type AiSkillDiagnostic,
  type AiSkillManifest,
  type AiSkillOrigin,
  type AiSkillSnapshot,
} from '../common/aiSkillTypes';
import type { AiContextSectionId } from '../common/aiContextTypes';

const maximumSkillSourceBytes = 256 * 1024;
const projectSkillRelativeRoot = path.join('.agents', 'skills');
const skillFileName = 'SKILL.md';

const validContextIds = new Set<AiContextSectionId>([
  'project',
  'scene',
  'selection',
  'workspace',
  'assets',
  'diagnostics',
  'viewport',
  'recentChanges',
]);
const validCapabilities = new Set<AiSkillCapability>(aiSkillCapabilities);
const identifierPattern = /^[a-z0-9][a-z0-9-]{0,63}$/;
const toolPattern = /^[a-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)*$/;
const semanticVersionPattern = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;
const knownMetadataKeys = new Set([
  'format',
  'formatVersion',
  'id',
  'name',
  'version',
  'description',
  'requires',
  'tools',
  'contexts',
]);

export type AiSkillProjectScope = {
  projectRoot: string;
  projectGuid: string;
};

export type AiSkillServiceOptions = {
  builtinRoot: string;
  project: () => AiSkillProjectScope | null;
};

type FrontMatterValue = string | string[];

type ParsedSkillMarkdown = {
  manifest: AiSkillManifest;
  instructions: string;
};

const scalarValue = (value: string): string => {
  const trimmed = value.trim();
  if (trimmed.length >= 2 && trimmed.startsWith('"') && trimmed.endsWith('"')) {
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (typeof parsed === 'string') return parsed;
    } catch {
      throw new Error('Double-quoted metadata values must be valid JSON strings');
    }
  }
  if (trimmed.length >= 2 && trimmed.startsWith("'") && trimmed.endsWith("'")) return trimmed.slice(1, -1);
  return trimmed;
};

const requireScalar = (metadata: Map<string, FrontMatterValue>, key: string): string => {
  const value = metadata.get(key);
  if (typeof value !== 'string' || !value.trim()) throw new Error(`Skill metadata '${key}' is required`);
  return value.trim();
};

const optionalArray = (metadata: Map<string, FrontMatterValue>, key: string): string[] => {
  const value = metadata.get(key);
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error(`Skill metadata '${key}' must be a list`);
  const result = value.map((entry) => entry.trim());
  if (result.some((entry) => !entry)) throw new Error(`Skill metadata '${key}' cannot contain empty entries`);
  if (new Set(result).size !== result.length) throw new Error(`Skill metadata '${key}' cannot contain duplicates`);
  return result;
};

const parseFrontMatter = (source: string): { metadata: Map<string, FrontMatterValue>; body: string } => {
  const normalized = source.replaceAll('\r\n', '\n');
  const lines = normalized.split('\n');
  if (lines[0] !== '---') throw new Error('Skill Markdown must start with YAML front matter');
  const end = lines.findIndex((line, index) => index > 0 && line === '---');
  if (end < 0) throw new Error('Skill Markdown front matter is not terminated');

  const metadata = new Map<string, FrontMatterValue>();
  let listKey: string | null = null;
  for (const rawLine of lines.slice(1, end)) {
    if (!rawLine.trim()) continue;
    const listItem = rawLine.match(/^\s+-\s+(.+)$/);
    if (listItem) {
      if (!listKey) throw new Error('Skill metadata list item has no parent field');
      const current = metadata.get(listKey);
      if (!Array.isArray(current)) throw new Error(`Skill metadata '${listKey}' must be a list`);
      current.push(scalarValue(listItem[1]));
      continue;
    }

    const field = rawLine.match(/^([A-Za-z][A-Za-z0-9]*):(?:\s*(.*))?$/);
    if (!field) throw new Error(`Unsupported skill metadata syntax: ${rawLine.trim()}`);
    const [, key, rawValue = ''] = field;
    if (!knownMetadataKeys.has(key)) throw new Error(`Unknown skill metadata field '${key}'`);
    if (metadata.has(key)) throw new Error(`Duplicate skill metadata field '${key}'`);
    if (rawValue.trim()) {
      metadata.set(key, scalarValue(rawValue));
      listKey = null;
    } else {
      metadata.set(key, []);
      listKey = key;
    }
  }

  const body = lines.slice(end + 1).join('\n').trim();
  if (!body) throw new Error('Skill Markdown must include instructions after the front matter');
  return { metadata, body };
};

export const parseAiSkillMarkdown = (source: string): ParsedSkillMarkdown => {
  const { metadata, body } = parseFrontMatter(source);
  if (requireScalar(metadata, 'format') !== AI_SKILL_FORMAT)
    throw new Error(`Skill format must be '${AI_SKILL_FORMAT}'`);
  const formatVersion = Number.parseInt(requireScalar(metadata, 'formatVersion'), 10);
  if (formatVersion !== AI_SKILL_FORMAT_VERSION)
    throw new Error(`Unsupported skill format version ${String(formatVersion)}`);

  const id = requireScalar(metadata, 'id');
  if (!identifierPattern.test(id)) throw new Error('Skill ID must use lowercase letters, numbers, and hyphens');
  const name = requireScalar(metadata, 'name');
  const version = requireScalar(metadata, 'version');
  if (!semanticVersionPattern.test(version)) throw new Error('Skill version must be semantic');
  const description = requireScalar(metadata, 'description');

  const requiredCapabilities = optionalArray(metadata, 'requires');
  const unknownCapability = requiredCapabilities.find(
    (capability) => !validCapabilities.has(capability as AiSkillCapability),
  );
  if (unknownCapability) throw new Error(`Skill requires unknown capability '${unknownCapability}'`);

  const tools = optionalArray(metadata, 'tools');
  const malformedTool = tools.find((tool) => !toolPattern.test(tool));
  if (malformedTool) throw new Error(`Skill tool declaration '${malformedTool}' is malformed`);

  const contexts = optionalArray(metadata, 'contexts');
  const unknownContext = contexts.find((context) => !validContextIds.has(context as AiContextSectionId));
  if (unknownContext) throw new Error(`Skill declares unknown context '${unknownContext}'`);

  return {
    manifest: {
      format: AI_SKILL_FORMAT,
      formatVersion: AI_SKILL_FORMAT_VERSION,
      id,
      name,
      version,
      description,
      requiredCapabilities: requiredCapabilities as AiSkillCapability[],
      tools,
      contexts: contexts as AiContextSectionId[],
    },
    instructions: body,
  };
};

const isContainedPath = (root: string, candidate: string): boolean => {
  const relative = path.relative(root, candidate);
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
};

export class AiSkillService {
  private revision = 0;
  private scopeKey = '';
  private current: AiSkillSnapshot | null = null;

  constructor(private readonly options: AiSkillServiceOptions) {}

  snapshot(force = false): AiSkillSnapshot {
    const project = this.options.project();
    const nextScopeKey = project ? `${project.projectGuid}:${path.resolve(project.projectRoot)}` : '<no-project>';
    if (!force && this.current && nextScopeKey === this.scopeKey) return this.current;

    const skills: AiSkill[] = [];
    const diagnostics: AiSkillDiagnostic[] = [];
    const ids = new Set<string>();
    this.scanRoot(this.options.builtinRoot, 'builtin', null, skills, diagnostics, ids, true);

    if (project) {
      const projectRoot = path.resolve(project.projectRoot);
      const skillRoot = path.join(projectRoot, projectSkillRelativeRoot);
      if (fs.existsSync(skillRoot)) {
        try {
          const realProjectRoot = fs.realpathSync(projectRoot);
          const realSkillRoot = fs.realpathSync(skillRoot);
          if (!isContainedPath(realProjectRoot, realSkillRoot)) throw new Error('Project skill root resolves outside the project');
          this.scanRoot(skillRoot, 'project', project.projectGuid, skills, diagnostics, ids, false);
        } catch (error) {
          diagnostics.push({
            origin: 'project',
            path: skillRoot,
            message: error instanceof Error ? error.message : String(error),
          });
        }
      }
    }

    this.scopeKey = nextScopeKey;
    this.current = {
      revision: ++this.revision,
      projectGuid: project?.projectGuid ?? null,
      skills,
      diagnostics,
    };
    return this.current;
  }

  invalidate(): void {
    this.current = null;
    this.scopeKey = '';
  }

  private scanRoot(
    root: string,
    origin: AiSkillOrigin,
    projectGuid: string | null,
    skills: AiSkill[],
    diagnostics: AiSkillDiagnostic[],
    ids: Set<string>,
    required: boolean,
  ): void {
    if (!root || !fs.existsSync(root)) {
      if (required)
        diagnostics.push({ origin, path: root, message: 'Built-in skill root is unavailable' });
      return;
    }

    let realRoot = '';
    try {
      if (!fs.statSync(root).isDirectory()) throw new Error('Skill root must be a directory');
      realRoot = fs.realpathSync(root);
    } catch (error) {
      diagnostics.push({ origin, path: root, message: error instanceof Error ? error.message : String(error) });
      return;
    }

    const entries = fs
      .readdirSync(root, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .sort((left, right) => left.name.localeCompare(right.name));

    for (const entry of entries) {
      const skillRoot = path.join(root, entry.name);
      const skillPath = path.join(skillRoot, skillFileName);
      try {
        const realSkillRoot = fs.realpathSync(skillRoot);
        if (!isContainedPath(realRoot, realSkillRoot)) throw new Error('Skill directory resolves outside its skill root');
        if (!fs.existsSync(skillPath) || !fs.statSync(skillPath).isFile()) throw new Error(`Skill directory is missing ${skillFileName}`);
        const realSkillPath = fs.realpathSync(skillPath);
        if (!isContainedPath(realSkillRoot, realSkillPath)) throw new Error(`${skillFileName} resolves outside its skill directory`);
        if (fs.statSync(realSkillPath).size > maximumSkillSourceBytes)
          throw new Error(`${skillFileName} exceeds the ${maximumSkillSourceBytes} byte limit`);

        const parsed = parseAiSkillMarkdown(fs.readFileSync(realSkillPath, 'utf8'));
        if (parsed.manifest.id !== entry.name)
          throw new Error(`Skill ID '${parsed.manifest.id}' must match directory '${entry.name}'`);
        if (ids.has(parsed.manifest.id))
          throw new Error(`Skill ID '${parsed.manifest.id}' duplicates an already loaded skill and was ignored`);
        ids.add(parsed.manifest.id);
        skills.push({
          ...parsed,
          origin,
          root: skillRoot,
          filePath: skillPath,
          projectGuid,
        });
      } catch (error) {
        diagnostics.push({
          origin,
          path: skillPath,
          message: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }
}
