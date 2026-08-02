import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { ProjectService } from './projectService';

const temporaryRoots: string[] = [];
const projectToolName = process.platform === 'win32' ? 'arc-project.exe' : 'arc-project';
const projectToolPath = [
  path.resolve(import.meta.dirname, '../../../out/build/default/tools/project_cli/RelWithDebInfo', projectToolName),
  path.resolve(import.meta.dirname, '../../../out/build/default/tools/project_cli', projectToolName),
].find((candidate) => fs.existsSync(candidate)) ?? '';
const templatesRoot = path.resolve(import.meta.dirname, '../../../templates');
const nativeProjectAuthority = { projectToolPath, templatesRoot };
const temporary = () => {
  const value = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-project-service-'));
  temporaryRoots.push(value);
  return value;
};

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('ProjectService', () => {
  it('creates an exact-version project and opens it through the native host', async () => {
    const root = temporary();
    const commands: Array<{ type: string; payload: Record<string, unknown> }> = [];
    const service = new ProjectService({
      ...nativeProjectAuthority,
      userDataPath: path.join(root, 'user'),
      currentEngineVersion: '1.2.3',
      currentEditorPath: 'arc-editor',
      host: {
        connected: true,
        error: '',
        command: async (type, payload = {}) => {
          commands.push({ type, payload });
          return { succeeded: true };
        },
      },
    });
    const created = service.create({ name: 'Alpine', destination: path.join(root, 'project'), template: 'mountain' });
    expect(created.succeeded, created.error).toBe(true);
    expect(created.project?.descriptor.engineVersion).toBe('1.2.3');
    const opened = await service.open(created.project!.descriptorPath);
    expect(opened.succeeded).toBe(true);
    expect(commands[0]).toMatchObject({
      type: 'project.open',
      payload: { name: 'Alpine', readOnly: false },
    });
    expect(service.snapshot().recentProjects[0].guid).toBe(created.project?.descriptor.guid);
  });

  it('requires explicit read-only mode for a newer project', async () => {
    const root = temporary();
    const descriptor = path.join(root, 'Future.arcproject');
    fs.writeFileSync(
      descriptor,
      JSON.stringify({
        format: 'arc-project',
        formatVersion: 1,
        guid: '00000000-0000-4000-8000-000000000001',
        name: 'Future',
        engineVersion: '9.0.0',
        assetRoots: ['assets'],
        startupScenes: [],
        modules: [],
        extensions: [],
        settings: {},
      }),
    );
    const service = new ProjectService({
      ...nativeProjectAuthority,
      userDataPath: path.join(root, 'user'),
      currentEngineVersion: '1.0.0',
      currentEditorPath: 'arc-editor',
      host: { connected: true, error: '', command: async () => ({ succeeded: true }) },
    });
    expect((await service.open(descriptor)).succeeded).toBe(false);
    expect((await service.open(descriptor, { readOnly: true })).project?.writable).toBe(false);
  });

  it('requires explicit upgrade mode and keeps a validated descriptor backup', async () => {
    const root = temporary();
    fs.mkdirSync(path.join(root, 'assets'));
    const descriptor = path.join(root, 'Legacy.arcproject');
    fs.writeFileSync(
      descriptor,
      JSON.stringify({
        format: 'arc-project',
        formatVersion: 1,
        guid: '00000000-0000-4000-8000-000000000002',
        name: 'Legacy',
        engineVersion: '1.0.0',
        assetRoots: ['assets'],
        startupScenes: [],
        modules: [],
        extensions: [],
        settings: {},
      }),
    );
    const service = new ProjectService({
      ...nativeProjectAuthority,
      userDataPath: path.join(root, 'user'),
      currentEngineVersion: '2.0.0',
      currentEditorPath: 'arc-editor',
      host: { connected: true, error: '', command: async () => ({ succeeded: true }) },
    });

    expect((await service.open(descriptor)).succeeded).toBe(false);
    const upgraded = await service.open(descriptor, { upgrade: true });
    expect(upgraded.succeeded, upgraded.error).toBe(true);
    expect(upgraded.project?.descriptor.engineVersion).toBe('2.0.0');
    const backup = JSON.parse(fs.readFileSync(`${descriptor}.v1.bak`, 'utf8')) as { engineVersion: string };
    expect(backup.engineVersion).toBe('1.0.0');
  });

  it('clones a local project into an existing empty destination without nesting it', async () => {
    const root = temporary();
    const host = { connected: true, error: '', command: async () => ({ succeeded: true }) };
    const service = new ProjectService({
      ...nativeProjectAuthority,
      userDataPath: path.join(root, 'user'),
      currentEngineVersion: '1.2.3',
      currentEditorPath: 'arc-editor',
      host,
    });
    const source = path.join(root, 'source');
    const created = service.create({ name: 'Source', destination: source });
    expect(created.succeeded, created.error).toBe(true);
    const destination = path.join(root, 'clone');
    fs.mkdirSync(destination);

    const cloned = await service.clone({ source, destination });

    expect(cloned.succeeded).toBe(true);
    expect(cloned.project?.projectRoot).toBe(destination);
    expect(fs.existsSync(path.join(destination, 'Source.arcproject'))).toBe(true);
    expect(fs.existsSync(path.join(destination, 'source', 'Source.arcproject'))).toBe(false);
  });

  it('rejects cloning a local project into its own subtree', async () => {
    const root = temporary();
    const service = new ProjectService({
      ...nativeProjectAuthority,
      userDataPath: path.join(root, 'user'),
      currentEngineVersion: '1.2.3',
      currentEditorPath: 'arc-editor',
      host: { connected: true, error: '', command: async () => ({ succeeded: true }) },
    });
    const source = path.join(root, 'source');
    const created = service.create({ name: 'Source', destination: source });
    expect(created.succeeded, created.error).toBe(true);

    const cloned = await service.clone({ source, destination: path.join(source, 'copy') });

    expect(cloned.succeeded).toBe(false);
    expect(cloned.error).toContain('descendants');
    expect(fs.existsSync(path.join(source, 'Source.arcproject'))).toBe(true);
  });
});
