import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { RecoveryGeneration } from '../common/editorWorkflowTypes';
import { RecoveryService } from './recoveryService';

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('RecoveryService', () => {
  it('reports unclean generations and restores through the native recovery command', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-recovery-'));
    roots.push(root);
    const projectRoot = path.join(root, 'project-guid');
    const recoveryPath = path.join(projectRoot, 'scene', 'one.arcscene');
    fs.mkdirSync(path.dirname(recoveryPath), { recursive: true });
    fs.writeFileSync(recoveryPath, '{}');
    const generation: RecoveryGeneration = {
      id: 'one',
      projectGuid: 'project-guid',
      documentGuid: 'scene',
      documentName: 'Scene',
      originalPath: 'assets/scenes/Scene.arcscene',
      recoveryPath,
      createdAt: new Date().toISOString(),
      historyRevision: 4,
      sceneRevision: 9,
      size: 2,
    };
    fs.writeFileSync(path.join(projectRoot, 'index.json'), JSON.stringify([generation]));
    fs.writeFileSync(path.join(projectRoot, 'heartbeat'), new Date().toISOString());
    fs.writeFileSync(path.join(projectRoot, 'clean'), new Date(0).toISOString());
    const commands: Array<{ type: string; payload?: Record<string, unknown> }> = [];
    const service = new RecoveryService(root, {
      query: async () => ({ succeeded: true }),
      command: async (type, payload) => {
        commands.push({ type, payload });
        return { succeeded: true };
      },
    });
    expect(service.snapshot('project-guid').uncleanShutdown).toBe(true);
    await service.restore('one');
    expect(commands[0]).toEqual({
      type: 'scene.openRecovery',
      payload: { path: recoveryPath, originalPath: 'assets/scenes/Scene.arcscene' },
    });
    expect(service.discard('one')).toBe(true);
    expect(service.snapshot('project-guid').generations).toHaveLength(0);
  });

  it('marks the previous project clean when the workspace switches normally', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-recovery-'));
    roots.push(root);
    const service = new RecoveryService(root, {
      query: async () => ({ succeeded: true }),
      command: async () => ({ succeeded: true }),
    });
    const project = (guid: string) =>
      ({
        descriptor: { guid },
      }) as Parameters<RecoveryService['start']>[0];

    service.start(project('first'));
    service.start(project('second'));

    expect(service.snapshot('first').uncleanShutdown).toBe(false);
    expect(service.snapshot('second').uncleanShutdown).toBe(true);
    service.stop(true);
    expect(service.snapshot('second').uncleanShutdown).toBe(false);
  });
});
