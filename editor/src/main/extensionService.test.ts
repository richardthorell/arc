import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { ArcProjectCandidate } from '../common/projectTypes';
import { ExtensionService } from './extensionService';

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('ExtensionService', () => {
  it('discovers compatible manifests and grants only read capabilities by default', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-extensions-'));
    roots.push(root);
    const extensionRoot = path.join(root, 'extensions', 'sample');
    fs.mkdirSync(extensionRoot, { recursive: true });
    fs.writeFileSync(path.join(extensionRoot, 'main.js'), 'export {};\n');
    fs.writeFileSync(
      path.join(extensionRoot, 'arc-extension.json'),
      JSON.stringify({
        format: 'arc-extension',
        formatVersion: 1,
        id: 'arc.sample',
        name: 'Sample',
        version: '1.0.0',
        engineVersion: '1.2.3',
        main: 'main.js',
        activationEvents: ['onProject'],
        capabilities: ['scene.read', 'scene.mutate'],
      }),
    );
    const project = {
      projectRoot: root,
      descriptor: { extensions: ['extensions/sample'] },
    } as ArcProjectCandidate;
    const service = new ExtensionService(() => project, '1.2.3');
    const snapshot = service.snapshot();
    expect(snapshot.extensions[0].enabled).toBe(true);
    expect(snapshot.extensions[0].grantedCapabilities).toEqual(['scene.read']);
  });

  it('rejects extension entry points that escape the extension root', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-extensions-'));
    roots.push(root);
    const extensionRoot = path.join(root, 'extension');
    fs.mkdirSync(extensionRoot);
    fs.writeFileSync(
      path.join(extensionRoot, 'arc-extension.json'),
      JSON.stringify({
        format: 'arc-extension',
        formatVersion: 1,
        id: 'arc.invalid',
        name: 'Invalid',
        version: '1.0.0',
        engineVersion: '1',
        main: '../outside.js',
        capabilities: [],
      }),
    );
    const project = { projectRoot: root, descriptor: { extensions: ['extension'] } } as ArcProjectCandidate;
    const extension = new ExtensionService(() => project, '1').snapshot().extensions[0];
    expect(extension.enabled).toBe(false);
    expect(extension.diagnostics[0]).toContain('entry point');
  });

  it('isolates invalid configured paths instead of failing extension discovery', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-extensions-'));
    roots.push(root);
    const project = {
      projectRoot: root,
      descriptor: { extensions: ['../outside', 'extensions/missing'] },
    } as ArcProjectCandidate;
    const snapshot = new ExtensionService(() => project, '1.0.0').snapshot();
    expect(snapshot.extensions).toHaveLength(2);
    expect(snapshot.extensions.every((extension) => !extension.enabled)).toBe(true);
    expect(snapshot.extensions[0].diagnostics[0]).toContain('escapes');
    expect(snapshot.extensions[1].diagnostics[0]).toContain('does not exist');
  });

  it('does not discover project code when extensions are disabled', () => {
    const project = { projectRoot: 'unused', descriptor: { extensions: ['extension'] } } as ArcProjectCandidate;
    expect(
      new ExtensionService(
        () => project,
        '1.0.0',
        () => false,
      ).snapshot().extensions,
    ).toEqual([]);
  });
});
