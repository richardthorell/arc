import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';

import type { ArcProjectCandidate } from '../common/projectTypes';
import { analyzeExternalModel, importExternalModel } from './externalModelImport';

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

  it('discovers glTF buffers and textures and copies only selected dependencies', () => {
    const project = makeProject();
    const sourceRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-external-gltf-source-'));
    temporaryRoots.push(sourceRoot);
    fs.mkdirSync(path.join(sourceRoot, 'textures'), { recursive: true });
    const source = path.join(sourceRoot, 'tree.gltf');
    fs.writeFileSync(
      source,
      JSON.stringify({
        buffers: [{ uri: 'tree.bin' }],
        images: [{ uri: 'textures/bark.png' }, { uri: 'data:image/png;base64,AAAA' }],
      }),
    );
    fs.writeFileSync(path.join(sourceRoot, 'tree.bin'), 'buffer-data');
    fs.writeFileSync(path.join(sourceRoot, 'textures', 'bark.png'), 'texture-data');

    const plan = analyzeExternalModel(source);
    expect(plan.dependencies.map(({ path, kind, exists }) => ({ path, kind, exists }))).toEqual([
      { path: 'textures/bark.png', kind: 'texture', exists: true },
      { path: 'tree.bin', kind: 'buffer', exists: true },
    ]);

    const imported = importExternalModel(source, project, 'Content/Trees', ['textures/bark.png']);
    expect(imported.path).toBe('Content/Trees/tree.gltf');
    expect(imported.importedDependencies).toEqual(['Content/Trees/textures/bark.png']);
    expect(fs.readFileSync(path.join(project.projectRoot, 'Content/Trees/textures/bark.png'), 'utf8')).toBe(
      'texture-data',
    );
    expect(fs.existsSync(path.join(project.projectRoot, 'Content/Trees/tree.bin'))).toBe(false);
  });

  it('discovers OBJ material libraries and textures referenced by the MTL', () => {
    const project = makeProject();
    const sourceRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-external-obj-source-'));
    temporaryRoots.push(sourceRoot);
    fs.mkdirSync(path.join(sourceRoot, 'textures'), { recursive: true });
    const source = path.join(sourceRoot, 'crate.obj');
    fs.writeFileSync(source, 'mtllib crate.mtl\nv 0 0 0\n');
    fs.writeFileSync(path.join(sourceRoot, 'crate.mtl'), 'newmtl crate\nmap_Kd textures/crate.png\n');
    fs.writeFileSync(path.join(sourceRoot, 'textures', 'crate.png'), 'texture-data');

    const plan = analyzeExternalModel(source);
    expect(plan.dependencies.map(({ path, kind }) => ({ path, kind }))).toEqual([
      { path: 'crate.mtl', kind: 'material' },
      { path: 'textures/crate.png', kind: 'texture' },
    ]);

    const imported = importExternalModel(
      source,
      project,
      'Content/Props',
      plan.dependencies.map((dependency) => dependency.path),
    );
    expect(imported.importedDependencies).toEqual(['Content/Props/crate.mtl', 'Content/Props/textures/crate.png']);
  });

  it('reports missing referenced files without attempting to copy them', () => {
    const sourceRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-external-missing-source-'));
    temporaryRoots.push(sourceRoot);
    const source = path.join(sourceRoot, 'tree.gltf');
    fs.writeFileSync(source, JSON.stringify({ images: [{ uri: 'missing.png' }] }));

    expect(analyzeExternalModel(source).dependencies).toEqual([
      expect.objectContaining({ path: 'missing.png', kind: 'texture', exists: false }),
    ]);
  });

  it('rejects a destination outside the configured content root', () => {
    const project = makeProject();
    const source = path.join(project.projectRoot, 'hero.glb');
    fs.writeFileSync(source, 'model-data');

    expect(() => importExternalModel(source, project, 'Saved/Models')).toThrow('content folder');
  });
});
