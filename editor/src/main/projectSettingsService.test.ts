import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ArcProjectCandidate } from '../common/projectTypes';
import { ProjectSettingsService } from './projectSettingsService';

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});
const capabilities = {
  version: 1,
  targets: ['windows-x64-vulkan', 'android-arm64-vulkan'],
  textureOutputs: ['bc', 'astc'],
  textureQualities: ['balanced'],
  androidMinimumSdk: 26,
  androidCompileSdk: 35,
};
function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-project-settings-'));
  roots.push(root);
  const descriptorPath = path.join(root, 'Game.arcproject');
  const raw = {
    guid: 'project',
    name: 'Game',
    engineVersion: '0.1.0',
    buildConfigurations: ['Shipping'],
    extension: { untouched: 42 },
    package: {
      applicationName: 'Game',
      companyName: '',
      output: 'Build/Packages',
      regionChunks: true,
      extension: 'retained',
    },
    cookProfiles: [
      {
        id: 'release',
        extension: 12,
        platform: 'windows',
        architecture: 'x86_64',
        renderer: 'vulkan',
        api: '1.2',
        configuration: 'Shipping',
        textures: { outputs: ['bc'], quality: 'balanced', extension: true },
      },
    ],
  };
  fs.writeFileSync(descriptorPath, JSON.stringify(raw));
  let project = {
    descriptorPath,
    projectRoot: root,
    writable: true,
    descriptor: raw,
  } as unknown as ArcProjectCandidate;
  const query = vi.fn(async (_tool: string, args: string[]): Promise<unknown> =>
    args[0] === 'settings' ? capabilities : { succeeded: true },
  );
  const workflowBusy = vi.fn(() => false);
  const publish = vi.fn((value: Record<string, unknown>) => {
    project = { ...project, descriptor: value as unknown as ArcProjectCandidate['descriptor'] };
  });
  const service = new ProjectSettingsService(
    () => project,
    () => 'tool',
    workflowBusy,
    publish,
    query,
  );
  const apply = async (changes: Record<string, unknown>) => {
    const before = await service.read();
    return service.apply({ catalogVersion: 1, projectGuid: before.projectGuid, revision: before.revision, changes });
  };
  return {
    root,
    raw,
    descriptorPath,
    service,
    query,
    publish,
    workflowBusy,
    apply,
    replace: () => {
      project = { ...project };
    },
    readOnly: () => {
      project.writable = false;
    },
  };
}
describe('Project Settings persistence', () => {
  it('returns settings without waiting for native option discovery', async () => {
    const value = fixture();
    let complete!: (value: unknown) => void;
    value.query.mockImplementation(
      () =>
        new Promise((resolve) => {
          complete = resolve;
        }),
    );
    const snapshot = await value.service.read();
    expect(snapshot.values.name).toBe('Game');
    expect(snapshot.capabilitiesPending).toBe(true);
    complete(capabilities);
    await Promise.resolve();
    expect((await value.service.read()).capabilities.targets).toContain('android-arm64-vulkan');
    expect(value.query).toHaveBeenCalledOnce();
  });
  it('rejects a descriptor replaced with a different project identity', async () => {
    const value = fixture();
    fs.writeFileSync(value.descriptorPath, JSON.stringify({ ...value.raw, guid: 'replacement' }));
    await expect(value.service.read()).rejects.toThrow('Project identity changed');
    expect(value.publish).not.toHaveBeenCalled();
  });
  it('validates a same-root temporary descriptor and atomically preserves unrelated/extension values', async () => {
    const value = fixture();
    const result = await value.apply({ name: 'Renamed', 'package.companyName': 'Studio' });
    expect(result.succeeded).toBe(true);
    const saved = JSON.parse(fs.readFileSync(value.descriptorPath, 'utf8'));
    expect(saved).toMatchObject({
      guid: 'project',
      name: 'Renamed',
      extension: { untouched: 42 },
      package: { companyName: 'Studio', extension: 'retained' },
    });
    const validation = value.query.mock.calls.find(([, args]) => args[0] === 'validate')!;
    expect(path.dirname(validation[1][2])).toBe(value.root);
    expect(fs.existsSync(validation[1][2])).toBe(false);
    expect(value.publish).toHaveBeenCalledOnce();
    expect(result.snapshot?.values.name).toBe('Renamed');
  });
  it('preserves extension metadata in profile and texture records', async () => {
    const value = fixture();
    const profile = { ...value.raw.cookProfiles[0], textures: { outputs: ['astc'], quality: 'balanced' } };
    delete (profile as { extension?: number }).extension;
    expect((await value.apply({ cookProfiles: [profile] })).succeeded).toBe(true);
    expect(JSON.parse(fs.readFileSync(value.descriptorPath, 'utf8')).cookProfiles[0]).toMatchObject({
      extension: 12,
      textures: { extension: true, outputs: ['astc'] },
    });
  });
  it.each([
    { guid: 'other' },
    { name: '' },
    { 'platformSettings.android.versionCode': 0 },
    { 'paths.content': '../elsewhere' },
  ])('rejects invalid/unauthorized changes without mutation %j', async (changes) => {
    const value = fixture();
    const before = fs.readFileSync(value.descriptorPath, 'utf8');
    expect((await value.apply(changes)).succeeded).toBe(false);
    expect(fs.readFileSync(value.descriptorPath, 'utf8')).toBe(before);
    expect(value.publish).not.toHaveBeenCalled();
  });
  it.each(['native', 'external', 'project', 'workflow'])(
    'does not publish when %s changes during validation',
    async (failure) => {
      const value = fixture();
      const before = fs.readFileSync(value.descriptorPath, 'utf8');
      value.query.mockImplementation(async (_tool, args) => {
        if (args[0] === 'settings') return capabilities;
        if (failure === 'native') throw new Error('Invalid project settings');
        if (failure === 'external') fs.writeFileSync(value.descriptorPath, before + '\n');
        if (failure === 'project') value.replace();
        if (failure === 'workflow') value.workflowBusy.mockReturnValue(true);
        return { succeeded: true };
      });
      expect((await value.apply({ name: 'Rejected' })).succeeded).toBe(false);
      expect(fs.readFileSync(value.descriptorPath, 'utf8').trim()).toBe(before);
      expect(value.publish).not.toHaveBeenCalled();
      expect(fs.readdirSync(value.root)).toEqual(['Game.arcproject']);
    },
  );
  it('rejects stale catalog/revision and read-only projects', async () => {
    const value = fixture();
    const snapshot = await value.service.read();
    const request = {
      catalogVersion: 1,
      projectGuid: snapshot.projectGuid,
      revision: snapshot.revision,
      changes: { name: 'Other' },
    };
    expect((await value.service.apply({ ...request, catalogVersion: 99 })).succeeded).toBe(false);
    value.replace();
    expect((await value.service.apply(request)).succeeded).toBe(false);
    value.readOnly();
    expect((await value.apply({ name: 'Other' })).succeeded).toBe(false);
  });
});
