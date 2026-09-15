import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { ProjectAssetWorkspace } from './projectAssetWorkspace';

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) fs.rmSync(directory, { recursive: true, force: true });
});

const project = (writable = true) => {
  const projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-agent-assets-'));
  temporaryDirectories.push(projectRoot);
  fs.mkdirSync(path.join(projectRoot, 'Content', 'Materials'), { recursive: true });
  return { projectRoot, assetRoots: ['Content'], writable };
};

describe('ProjectAssetWorkspace', () => {
  it('creates only new files below the primary content root and can roll them back', async () => {
    const context = project();
    const workspace = new ProjectAssetWorkspace(() => context);
    await workspace.create('Materials/Agent.arcmat', '{"version":4}\n');

    const target = path.join(context.projectRoot, 'Content', 'Materials', 'Agent.arcmat');
    expect(fs.readFileSync(target, 'utf8')).toBe('{"version":4}\n');
    await expect(workspace.create('Materials/Agent.arcmat', '{}')).rejects.toThrow(/already exists/);
    await workspace.remove('Materials/Agent.arcmat');
    expect(fs.existsSync(target)).toBe(false);
  });

  it('rejects traversal, read-only writes, and removal of pre-existing assets', async () => {
    const context = project(false);
    const workspace = new ProjectAssetWorkspace(() => context);
    await expect(workspace.create('Materials/Agent.arcmat', '{}')).rejects.toThrow(/read-only/);
    await expect(workspace.exists('../outside.arcmat')).rejects.toThrow(/relative/);

    context.writable = true;
    const existing = path.join(context.projectRoot, 'Content', 'Materials', 'Existing.arcmat');
    fs.writeFileSync(existing, '{}', 'utf8');
    await expect(workspace.remove('Materials/Existing.arcmat')).rejects.toThrow(/not created/);
    expect(fs.existsSync(existing)).toBe(true);
  });
});
