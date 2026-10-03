import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { loadAiInstructionSources } from './aiInstructionSourceService';

const roots: string[] = [];

const temporaryRoot = (): string => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-ai-instructions-'));
  roots.push(root);
  return root;
};

const skillSource = (id: string): string => `---
format: arc-skill
formatVersion: 1
id: ${id}
name: ${id}
version: 1.0.0
description: Inspect scene entities and assets
requires:
  - scene.read
  - asset.read
tools:
  - scene.overview
contexts:
  - scene
---

# ${id}

Inspect the relevant ARC scene state.
`;

const writeSkill = (root: string, id: string): void => {
  const directory = path.join(root, id);
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(path.join(directory, 'SKILL.md'), skillSource(id), 'utf8');
};

const writeProject = (root: string, guid: string): string => {
  const descriptorPath = path.join(root, 'project.arcproject');
  fs.mkdirSync(root, { recursive: true });
  fs.writeFileSync(descriptorPath, JSON.stringify({ format: 'arc-project', guid }), 'utf8');
  return descriptorPath;
};

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('loadAiInstructionSources', () => {
  it('loads built-in skills, active-project skills, and project AGENTS.md', () => {
    const root = temporaryRoot();
    const builtinRoot = path.join(root, 'builtin');
    const projectRoot = path.join(root, 'project');
    fs.mkdirSync(builtinRoot, { recursive: true });
    writeSkill(builtinRoot, 'scene-inspection');
    writeSkill(path.join(projectRoot, '.agents', 'skills'), 'project-scene-review');
    const descriptorPath = writeProject(projectRoot, 'project-guid');
    fs.writeFileSync(path.join(projectRoot, 'AGENTS.md'), 'Use metres and stable GUIDs.\n', 'utf8');

    const snapshot = loadAiInstructionSources(builtinRoot, {
      projectRoot,
      descriptorPath,
      projectGuid: 'project-guid',
    });

    expect(snapshot.projectGuid).toBe('project-guid');
    expect(snapshot.projectInstructions).toBe('Use metres and stable GUIDs.');
    expect(snapshot.skills.map((entry) => [entry.manifest.id, entry.origin])).toEqual([
      ['scene-inspection', 'builtin'],
      ['project-scene-review', 'project'],
    ]);
    expect(snapshot.diagnostics).toEqual([]);
    expect(snapshot.skills[0]).not.toHaveProperty('filePath');
    expect(snapshot.skills[0]).not.toHaveProperty('root');
  });

  it('rejects a mismatched project descriptor without leaking project instructions or skills', () => {
    const root = temporaryRoot();
    const builtinRoot = path.join(root, 'builtin');
    const projectRoot = path.join(root, 'project');
    fs.mkdirSync(builtinRoot, { recursive: true });
    writeSkill(builtinRoot, 'scene-inspection');
    writeSkill(path.join(projectRoot, '.agents', 'skills'), 'project-scene-review');
    const descriptorPath = writeProject(projectRoot, 'actual-guid');
    fs.writeFileSync(path.join(projectRoot, 'AGENTS.md'), 'Project-only instructions', 'utf8');

    const snapshot = loadAiInstructionSources(builtinRoot, {
      projectRoot,
      descriptorPath,
      projectGuid: 'different-guid',
    });

    expect(snapshot.projectGuid).toBeNull();
    expect(snapshot.projectInstructions).toBeUndefined();
    expect(snapshot.skills.map((entry) => entry.manifest.id)).toEqual(['scene-inspection']);
    expect(snapshot.diagnostics.map((entry) => entry.message).join('\n')).toContain('GUID does not match');
  });
});
