import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { AiSkillService, parseAiSkillMarkdown, type AiSkillProjectScope } from './aiSkillService';

const temporaryRoots: string[] = [];

const temporaryRoot = (): string => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-ai-skills-'));
  temporaryRoots.push(root);
  return root;
};

const skillSource = (
  id: string,
  options: { requires?: string[]; tools?: string[]; contexts?: string[]; extra?: string } = {},
): string => `---
format: arc-skill
formatVersion: 1
id: ${id}
name: ${id}
version: 1.0.0
description: Test skill ${id}
requires:
${(options.requires ?? ['scene.read']).map((entry) => `  - ${entry}`).join('\n')}
tools:
${(options.tools ?? ['scene.overview']).map((entry) => `  - ${entry}`).join('\n')}
contexts:
${(options.contexts ?? ['scene']).map((entry) => `  - ${entry}`).join('\n')}
${options.extra ?? ''}---

# ${id}

Use the declared context and tools without changing editor permissions.
`;

const writeSkill = (root: string, id: string, source = skillSource(id)): void => {
  const directory = path.join(root, id);
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(path.join(directory, 'SKILL.md'), source, 'utf8');
};

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('AiSkillService', () => {
  it('loads built-in and active-project skills without leaking project scope', () => {
    const root = temporaryRoot();
    const builtinRoot = path.join(root, 'builtin');
    const projectA = path.join(root, 'project-a');
    const projectB = path.join(root, 'project-b');
    fs.mkdirSync(builtinRoot, { recursive: true });
    fs.mkdirSync(projectA, { recursive: true });
    fs.mkdirSync(projectB, { recursive: true });
    writeSkill(builtinRoot, 'scene-inspection');
    writeSkill(path.join(projectA, '.agents', 'skills'), 'project-workflow');
    writeSkill(path.join(projectB, '.agents', 'skills'), 'other-workflow');

    let project: AiSkillProjectScope | null = { projectRoot: projectA, projectGuid: 'project-a-guid' };
    const service = new AiSkillService({ builtinRoot, project: () => project });

    const first = service.snapshot();
    expect(first.skills.map((skill) => skill.manifest.id)).toEqual(['scene-inspection', 'project-workflow']);
    expect(first.skills[0].origin).toBe('builtin');
    expect(first.skills[0].projectGuid).toBeNull();
    expect(first.skills[1].origin).toBe('project');
    expect(first.skills[1].projectGuid).toBe('project-a-guid');
    expect(first.diagnostics).toEqual([]);

    project = { projectRoot: projectB, projectGuid: 'project-b-guid' };
    const second = service.snapshot();
    expect(second.skills.map((skill) => skill.manifest.id)).toEqual(['scene-inspection', 'other-workflow']);
    expect(second.skills.some((skill) => skill.manifest.id === 'project-workflow')).toBe(false);

    project = null;
    const noProject = service.snapshot();
    expect(noProject.skills.map((skill) => skill.manifest.id)).toEqual(['scene-inspection']);
    expect(noProject.projectGuid).toBeNull();
  });

  it('rejects invalid skills and prevents project skills from shadowing built-ins', () => {
    const root = temporaryRoot();
    const builtinRoot = path.join(root, 'builtin');
    const projectRoot = path.join(root, 'project');
    fs.mkdirSync(builtinRoot, { recursive: true });
    fs.mkdirSync(projectRoot, { recursive: true });
    writeSkill(builtinRoot, 'scene-inspection');
    const projectSkills = path.join(projectRoot, '.agents', 'skills');
    writeSkill(projectSkills, 'scene-inspection');
    writeSkill(projectSkills, 'unsafe-skill', skillSource('unsafe-skill', { extra: 'permissions: all\n' }));

    const service = new AiSkillService({
      builtinRoot,
      project: () => ({ projectRoot, projectGuid: 'project-guid' }),
    });
    const snapshot = service.snapshot();

    expect(snapshot.skills.map((skill) => skill.manifest.id)).toEqual(['scene-inspection']);
    expect(snapshot.diagnostics).toHaveLength(2);
    expect(snapshot.diagnostics.map((diagnostic) => diagnostic.message).join('\n')).toContain('duplicates an already loaded skill');
    expect(snapshot.diagnostics.map((diagnostic) => diagnostic.message).join('\n')).toContain("Unknown skill metadata field 'permissions'");
  });

  it('validates versioned metadata, declared capabilities, tools, and contexts', () => {
    const parsed = parseAiSkillMarkdown(
      skillSource('renderer-diagnostics', {
        requires: ['viewport.read', 'diagnostics.read'],
        tools: ['viewport.debug', 'diagnostics.get'],
        contexts: ['viewport', 'diagnostics'],
      }),
    );
    expect(parsed.manifest.requiredCapabilities).toEqual(['viewport.read', 'diagnostics.read']);
    expect(parsed.manifest.tools).toEqual(['viewport.debug', 'diagnostics.get']);
    expect(parsed.manifest.contexts).toEqual(['viewport', 'diagnostics']);
    expect(parsed.instructions).toContain('without changing editor permissions');

    expect(() =>
      parseAiSkillMarkdown(skillSource('bad-capability', { requires: ['filesystem.unrestricted'] })),
    ).toThrow("unknown capability 'filesystem.unrestricted'");
    expect(() => parseAiSkillMarkdown(skillSource('bad-context', { contexts: ['secrets'] }))).toThrow(
      "unknown context 'secrets'",
    );
  });

  it('loads the shipped ARC workflow skill inventory', () => {
    const service = new AiSkillService({
      builtinRoot: path.resolve(process.cwd(), 'resources', 'ai-skills'),
      project: () => null,
    });
    const snapshot = service.snapshot(true);

    expect(snapshot.diagnostics).toEqual([]);
    expect(snapshot.skills.map((skill) => skill.manifest.id)).toEqual([
      'flow-authoring',
      'material-authoring',
      'play-workflows',
      'renderer-diagnostics',
      'scene-inspection',
      'terrain-workflows',
    ]);
  });
});
