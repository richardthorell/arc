import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { expect, it } from 'vitest';
import type { ArcProjectCandidate } from '../common/projectTypes';
import { CookWorkflowService } from './cookWorkflowService';
import { runWorkflowProcess } from './workflowProcess';

const tool = process.env.ARC_WORKFLOW_TEST_PROJECT_TOOL;
const cooker = process.env.ARC_WORKFLOW_TEST_COOKER;
it.skipIf(!tool || !cooker).each(['windows-x64-vulkan', 'android-arm64-vulkan'])(
  'cooks and verifies saved scene content through the real native workflow (%s)',
  async (profile) => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-cook-smoke-'));
    try {
      const projectRoot = path.join(root, 'Game');
      await runWorkflowProcess({
        executable: tool!,
        args: [
          'create',
          '--name',
          'Game',
          '--destination',
          projectRoot,
          '--template',
          'blank-3d',
          '--templates',
          path.resolve('..', 'templates'),
          '--engine',
          '0.1.0',
        ],
        cwd: root,
        signal: new AbortController().signal,
      });
      const descriptorPath = path.join(projectRoot, 'Game.arcproject');
      const descriptor = JSON.parse(fs.readFileSync(descriptorPath, 'utf8'));
      const sceneGuid = 'c09fb45a-3349-4ccb-a367-af470c330101';
      descriptor.defaultScene = { guid: sceneGuid, pathHint: 'Content/Smoke.arcscene' };
      fs.writeFileSync(descriptorPath, JSON.stringify(descriptor));
      const scene = path.join(projectRoot, 'Content', 'Smoke.arcscene');
      fs.writeFileSync(
        scene,
        JSON.stringify({
          format: 'arc.scene',
          formatVersion: 3,
          scene: { id: '7dc8bf4e-a087-4789-b72f-eb18f4500101', name: 'Smoke' },
          entities: [],
        }),
      );
      fs.copyFileSync(
        path.resolve('..', 'assets', 'fixtures', 'persistence_fixture.arcscene.arcmeta'),
        `${scene}.arcmeta`,
      );
      const project = { descriptorPath, projectRoot, descriptor, writable: true } as ArcProjectCandidate;
      const service = new CookWorkflowService(
        () => project,
        () => tool!,
        () => false,
        () => {},
        () => cooker!,
      );
      const request = {
        action: 'cook',
        profile,
        targetId: profile,
        destination: 'none',
        sessionId: '',
      } as const;
      const before = fs.readFileSync(scene, 'utf8');
      const first = await service.execute(request);
      expect(first.error).toBeUndefined();
      expect(first.state).toBe('succeeded');
      const second = await service.execute(request);
      expect(second.state).toBe('succeeded');
      expect(first.artifacts[0].path).not.toBe(second.artifacts[0].path);
      expect(fs.existsSync(first.artifacts[0].manifestPath)).toBe(true);
      expect(fs.readFileSync(scene, 'utf8')).toBe(before);
      const completion = service
        .snapshot()
        .output.join('')
        .split(/\r?\n/u)
        .flatMap((line) => {
          try {
            const value = JSON.parse(line);
            return value.event === 'cook.complete' ? [value] : [];
          } catch {
            return [];
          }
        });
      expect(completion).toHaveLength(1);
      expect(completion[0].cacheHits).toBeGreaterThan(0);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  },
  120_000,
);
