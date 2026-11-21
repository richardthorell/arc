import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { LivePlaySceneBaselineStore, MAX_LIVE_PLAY_SCENE_BYTES } from './livePlaySceneBaseline';

const roots: string[] = [];
const identity = {
  projectGuid: '11111111-1111-1111-1111-111111111111',
  buildId: 'dev-build-42',
  targetProfile: 'windows-default',
  sessionId: 'session-123',
  instanceId: 'runtime-456',
};

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-baseline-test-'));
  roots.push(root);
  const projectRoot = path.join(root, 'project');
  const tempRoot = path.join(root, 'private');
  fs.mkdirSync(projectRoot);
  fs.mkdirSync(tempRoot);
  return { root, projectRoot, tempRoot };
}

function scene(sceneGuid = '22222222-2222-2222-2222-222222222222') {
  return {
    format: 'arc.scene',
    formatVersion: 2,
    scene: { id: sceneGuid, name: 'Unsaved' },
    entities: [
      {
        id: '33333333-3333-3333-3333-333333333333',
        parent: null,
        components: {
          Name: { version: 1, value: 'Cube' },
          Transform: { version: 1, position: [4, 5, 6], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
          PointLight: { version: 1, intensity: 2 },
        },
      },
    ],
    dependencies: [],
  };
}

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('LivePlaySceneBaselineStore', () => {
  it('captures the native Authoring World without saving project files and preserves identity', async () => {
    const { projectRoot, tempRoot } = fixture();
    const original = path.join(projectRoot, 'Content', 'Level.arcscene');
    const command = vi.fn(async (type: string, payload: Record<string, unknown>) => {
      expect(type).toBe('scene.autosave');
      expect(payload.path).toContain('initial-scene.arcscene');
      fs.writeFileSync(payload.path as string, JSON.stringify(scene()));
      return {
        succeeded: true,
        payload: { sceneGuid: scene().scene.id, historyRevision: 9, sourceScenePath: original },
        sceneRevision: 18,
      };
    });
    const store = new LivePlaySceneBaselineStore({ command }, tempRoot);
    const captured = await store.prepare({ ...identity, projectRoot });
    expect(command).toHaveBeenCalledOnce();
    expect(captured.sceneGuid).toBe(scene().scene.id);
    expect(captured.sourceScenePath).toBe(original);
    expect(captured.sceneRevision).toBe(18);
    expect(captured.historyRevision).toBe(9);
    expect(captured.entityCount).toBe(1);
    expect(captured.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(fs.existsSync(original)).toBe(false);
    expect(captured.directory.startsWith(projectRoot)).toBe(false);
    const staged = JSON.parse(fs.readFileSync(captured.scenePath, 'utf8')) as ReturnType<typeof scene>;
    expect(staged.entities[0].components.Transform.position).toEqual([4, 5, 6]);
    expect(staged.entities[0].components.PointLight.intensity).toBe(2);
    expect(store.current()).toBe(captured);
    store.clear('not-this-session');
    expect(fs.existsSync(captured.scenePath)).toBe(true);
    store.clear(identity.sessionId);
    expect(store.current()).toBeNull();
    expect(fs.existsSync(captured.directory)).toBe(false);
  });

  it('rejects an invalid or mismatched native scene and cleans its staging directory', async () => {
    const { projectRoot, tempRoot } = fixture();
    const store = new LivePlaySceneBaselineStore(
      {
        command: async (_type, payload) => {
          fs.writeFileSync(payload.path as string, JSON.stringify(scene()));
          return {
            succeeded: true,
            payload: { sceneGuid: '44444444-4444-4444-4444-444444444444', historyRevision: 1 },
            sceneRevision: 2,
          };
        },
      },
      tempRoot,
    );
    await expect(store.prepare({ ...identity, projectRoot })).rejects.toThrow(/mismatched ARC scene/);
    expect(fs.readdirSync(tempRoot)).toHaveLength(0);
  });

  it('rejects oversized scene output without retaining temporary files', async () => {
    const { projectRoot, tempRoot } = fixture();
    const store = new LivePlaySceneBaselineStore(
      {
        command: async (_type, payload) => {
          fs.writeFileSync(payload.path as string, Buffer.alloc(MAX_LIVE_PLAY_SCENE_BYTES + 1));
          return {
            succeeded: true,
            payload: { sceneGuid: scene().scene.id, historyRevision: 1 },
            sceneRevision: 2,
          };
        },
      },
      tempRoot,
    );
    await expect(store.prepare({ ...identity, projectRoot })).rejects.toThrow(/16 MiB/);
    expect(fs.readdirSync(tempRoot)).toHaveLength(0);
  });

  it('cancels a pending host capture without publishing a stale session baseline', async () => {
    const { projectRoot, tempRoot } = fixture();
    let finish!: () => void;
    const pending = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const store = new LivePlaySceneBaselineStore(
      {
        command: async (_type, payload) => {
          await pending;
          fs.writeFileSync(payload.path as string, JSON.stringify(scene()));
          return {
            succeeded: true,
            payload: { sceneGuid: scene().scene.id, historyRevision: 1 },
            sceneRevision: 2,
          };
        },
      },
      tempRoot,
    );
    const capture = store.prepare({ ...identity, projectRoot });
    store.clear(identity.sessionId);
    finish();
    await expect(capture).rejects.toThrow(/cancelled/);
    expect(store.current()).toBeNull();
    expect(fs.readdirSync(tempRoot)).toHaveLength(0);
  });

  it('rejects a failed native save and prevents a second concurrent capture', async () => {
    const { projectRoot, tempRoot } = fixture();
    let finish!: () => void;
    const pending = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const store = new LivePlaySceneBaselineStore(
      {
        command: async () => {
          await pending;
          return { succeeded: false, error: 'invalid terrain arrays' };
        },
      },
      tempRoot,
    );
    const capture = store.prepare({ ...identity, projectRoot });
    await expect(store.prepare({ ...identity, projectRoot })).rejects.toThrow(/already active/);
    finish();
    await expect(capture).rejects.toThrow(/invalid terrain arrays/);
    expect(fs.readdirSync(tempRoot)).toHaveLength(0);
  });
});
