import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';

import type { ArcProjectCandidate } from '../common/projectTypes';
import { importExternalModel } from './externalModelImport';

const temporaryRoots: string[] = [];

const makeProject = (): ArcProjectCandidate => {
  const projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-external-model-'));
  temporaryRoots.push(projectRoot);
  return {
    descriptor: {
      format: 'arc-project',
      formatVersion: 2,
      guid: 'test-project',
      name: 'Test Project',
      engineVersion: 'dev',
      paths: {
        source: 'Source',
        content: 'Content',
        config: 'Config',
        plugins: 'Plugins',
        saved: 'Saved',
        intermediate: 'Intermediate',
        build: 'Build',
      },
      assetRoots: ['Content'],
      modules: [],
      plugins: [],
      defaultScene: null,
      startupScenes: [],
      targetPlatforms: [],
      toolchain: {
        compiler: '',
        minimumVersion: '',
        generator: '',
        architecture: '',
        cppStandard: 23,
      },
      buildConfigurations: [],
      renderer: { backend: 'vulkan', api: '1.3', quality: 'default' },
      cookProfiles: [],
      package: { applicationName: 'Test', companyName: 'ARC', output: 'Build', regionChunks: false },
      settings: { editor: '', renderer: '', input: '' },
    },
    descriptorPath: path.join(projectRoot, 'Test.arcproject'),
    projectRoot,
    compatibility: 'compatible',
    writable: true,
    diagnostics: [],
  };
};

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('external model import', () => {
  it('copies a model into the requested selected content folder without overwriting the source', () => {
    const project = makeProject();
    const sourceRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-external-model-source-'));
    temporaryRoots.push(sourceRoot);
    const source = path.join(sourceRoot, 'hero.glb');
    fs.writeFileSync(source, 'model-data');

    const imported = importExternalModel(source, project, 'Content/Characters/Hero');

    expect(imported.path).toBe('Content/Characters/Hero/hero.glb');
    expect(imported.sourcePath).toBe(path.join(project.projectRoot, imported.path));
    expect(fs.readFileSync(imported.sourcePath, 'utf8')).toBe('model-data');
    expect(fs.readFileSync(source, 'utf8')).toBe('model-data');
  });

  it('uses the configured project content root', () => {
    const project = makeProject();
    project.descriptor.paths.content = 'GameContent';
    const sourceRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-external-model-source-'));
    temporaryRoots.push(sourceRoot);
    const source = path.join(sourceRoot, 'hero.glb');
    fs.writeFileSync(source, 'model-data');

    const imported = importExternalModel(source, project, 'GameContent/Props');

    expect(imported.path).toBe('GameContent/Props/hero.glb');
  });

  it('rejects a destination outside the configured content root', () => {
    const project = makeProject();
    const source = path.join(project.projectRoot, 'hero.glb');
    fs.writeFileSync(source, 'model-data');

    expect(() => importExternalModel(source, project, 'Saved/Models')).toThrow('content folder');
  });
});
