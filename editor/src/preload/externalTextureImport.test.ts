import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';

import type { ArcProjectCandidate } from '../common/projectTypes';
import { importExternalTexture, isSupportedTexturePath } from './externalTextureImport';

const temporaryRoots: string[] = [];

const makeProject = (): ArcProjectCandidate => {
  const projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-external-texture-'));
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

describe('external texture import', () => {
  it('recognizes supported texture formats case-insensitively', () => {
    expect(isSupportedTexturePath('albedo.PNG')).toBe(true);
    expect(isSupportedTexturePath('mesh.glb')).toBe(false);
  });

  it('copies a texture into Content/Textures and preserves the original', () => {
    const project = makeProject();
    const sourceRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-external-source-'));
    temporaryRoots.push(sourceRoot);
    const source = path.join(sourceRoot, 'albedo.png');
    fs.writeFileSync(source, 'texture-data');

    const imported = importExternalTexture(source, project);

    expect(imported.path).toBe('Content/Textures/albedo.png');
    expect(fs.readFileSync(path.join(project.projectRoot, imported.path), 'utf8')).toBe('texture-data');
    expect(fs.readFileSync(source, 'utf8')).toBe('texture-data');
  });

  it('does not overwrite an existing project texture', () => {
    const project = makeProject();
    const sourceRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-external-source-'));
    temporaryRoots.push(sourceRoot);
    const source = path.join(sourceRoot, 'normal.png');
    fs.writeFileSync(source, 'new');
    const destinationDirectory = path.join(project.projectRoot, 'Content', 'Textures');
    fs.mkdirSync(destinationDirectory, { recursive: true });
    fs.writeFileSync(path.join(destinationDirectory, 'normal.png'), 'existing');

    const imported = importExternalTexture(source, project);

    expect(imported.path).toBe('Content/Textures/normal_1.png');
    expect(fs.readFileSync(path.join(destinationDirectory, 'normal.png'), 'utf8')).toBe('existing');
    expect(fs.readFileSync(path.join(project.projectRoot, imported.path), 'utf8')).toBe('new');
  });

  it('rejects imports into a read-only project', () => {
    const project = { ...makeProject(), writable: false };
    const source = path.join(project.projectRoot, 'texture.png');
    fs.writeFileSync(source, 'texture-data');

    expect(() => importExternalTexture(source, project)).toThrow('read-only');
  });
});
