import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ArcProjectCandidate } from '../common/projectTypes';
import { ProjectSettingsService } from './projectSettingsService';

const roots: string[] = [];
it('persists project information while retaining custom application names', async () => {
  const state = fixture();
  const result = await state.apply({ name: 'Renamed', description: 'First line\nSecond line', version: '1.2.3' });
  expect(result.succeeded).toBe(true);
  const saved = JSON.parse(fs.readFileSync(state.descriptorPath, 'utf8'));
  expect(saved).toMatchObject({
    name: 'Renamed',
    description: 'First line\nSecond line',
    version: '1.2.3',
    package: { applicationName: 'Renamed' },
  });
  saved.package.applicationName = 'Custom application';
  fs.writeFileSync(state.descriptorPath, JSON.stringify(saved));
  expect((await state.apply({ name: 'Another name' })).succeeded).toBe(true);
  expect(JSON.parse(fs.readFileSync(state.descriptorPath, 'utf8')).package.applicationName).toBe('Custom application');
});

it('rejects invalid project versions without changing the descriptor', async () => {
  const state = fixture();
  const before = fs.readFileSync(state.descriptorPath, 'utf8');
  expect((await state.apply({ version: '1.2', description: 'Must not persist' })).succeeded).toBe(false);
  expect(fs.readFileSync(state.descriptorPath, 'utf8')).toBe(before);
});
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
    distributionVariants: [{ id: 'preserved-channel', targetId: 'android-arm64-vulkan', extension: { future: true } }],
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
    () => true,
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
  it('migrates extra startup references to cook inclusion without changing editor startup or dropping metadata', async () => {
    const state = fixture();
    const first = { guid: 'first', expectedType: 'scene', pathHint: 'Content/First.arcscene' };
    const extra = { guid: 'extra', expectedType: 'scene', pathHint: 'Content/Extra.arcscene', extension: 42 };
    fs.writeFileSync(
      state.descriptorPath,
      JSON.stringify({ ...state.raw, defaultScene: first, startupScenes: [first, extra] }),
    );
    const read = await state.service.read();
    expect(read.values.startupScene).toEqual(first);
    expect(read.values.includedScenes).toEqual([extra]);
    expect((await state.apply({ startupScene: null })).succeeded).toBe(true);
    const saved = JSON.parse(fs.readFileSync(state.descriptorPath, 'utf8'));
    expect(saved.startupScenes).toEqual([]);
    expect(saved.includedScenes).toEqual([extra]);
    expect(saved.defaultScene).toEqual(first);
    expect(saved.startupScene).toBeUndefined();
  });
  it('publishes a draft thumbnail only on Apply and removes it explicitly', async () => {
    const state = fixture();
    const thumbnail = path.join(state.root, 'Config', 'ProjectThumbnail.webp');
    const image = Buffer.from('RIFF0000WEBPdraft');
    const before = await state.service.read();
    expect(fs.existsSync(thumbnail)).toBe(false);
    const result = await state.service.apply({
      catalogVersion: 1,
      projectGuid: before.projectGuid,
      revision: before.revision,
      changes: { thumbnail: 'Config/ProjectThumbnail.webp' },
      thumbnailUpload: `data:image/webp;base64,${image.toString('base64')}`,
    });
    expect(result.succeeded).toBe(true);
    expect(fs.readFileSync(thumbnail)).toEqual(image);
    expect(JSON.parse(fs.readFileSync(state.descriptorPath, 'utf8')).thumbnail).toBe('Config/ProjectThumbnail.webp');
    expect((await state.apply({ thumbnail: null })).succeeded).toBe(true);
    expect(fs.existsSync(thumbnail)).toBe(false);
  });

  it('keeps the existing image and descriptor when validation fails', async () => {
    const state = fixture();
    const thumbnail = path.join(state.root, 'Config', 'ProjectThumbnail.webp');
    fs.mkdirSync(path.dirname(thumbnail));
    fs.writeFileSync(thumbnail, 'previous');
    const before = await state.service.read();
    state.query.mockResolvedValueOnce({ succeeded: false });
    const result = await state.service.apply({
      catalogVersion: 1,
      projectGuid: before.projectGuid,
      revision: before.revision,
      changes: { thumbnail: 'Config/ProjectThumbnail.webp' },
      thumbnailUpload: `data:image/webp;base64,${Buffer.from('RIFF0000WEBPdraft').toString('base64')}`,
    });
    expect(result.succeeded).toBe(false);
    expect(fs.readFileSync(thumbnail, 'utf8')).toBe('previous');
    expect(JSON.parse(fs.readFileSync(state.descriptorPath, 'utf8')).thumbnail).toBeUndefined();
  });

  it('rolls back the thumbnail if descriptor publication fails', async () => {
    const state = fixture();
    const thumbnail = path.join(state.root, 'Config', 'ProjectThumbnail.webp');
    fs.mkdirSync(path.dirname(thumbnail));
    fs.writeFileSync(thumbnail, 'previous');
    const before = await state.service.read();
    const rename = fs.renameSync;
    const spy = vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
      if (to === state.descriptorPath) throw new Error('Publication failed');
      rename(from, to);
    });
    try {
      const result = await state.service.apply({
        catalogVersion: 1,
        projectGuid: before.projectGuid,
        revision: before.revision,
        changes: { thumbnail: 'Config/ProjectThumbnail.webp' },
        thumbnailUpload: `data:image/webp;base64,${Buffer.from('RIFF0000WEBPdraft').toString('base64')}`,
      });
      expect(result.succeeded).toBe(false);
      expect(fs.readFileSync(thumbnail, 'utf8')).toBe('previous');
      expect(JSON.parse(fs.readFileSync(state.descriptorPath, 'utf8')).thumbnail).toBeUndefined();
    } finally {
      spy.mockRestore();
    }
  });
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
      distributionVariants: value.raw.distributionVariants,
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
