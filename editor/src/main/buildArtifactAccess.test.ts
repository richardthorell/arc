import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { WorkflowArtifact } from '../common/projectWorkflowTypes';
import { resolveBuildArtifactPath } from './buildArtifactAccess';

const roots: string[] = [];
const fixture = () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-artifact-'));
  roots.push(root);
  const file = path.join(root, 'Build', 'Packages', 'Game.apk');
  const manifest = path.join(root, 'Build', 'Packages', 'arc-distribution.json');
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, 'apk');
  fs.writeFileSync(manifest, '{}');
  const artifacts: WorkflowArtifact[] = [
    {
      kind: 'package',
      path: 'Build/Packages/Game.apk',
      manifestPath: 'Build/Packages/arc-distribution.json',
      buildId: 'run-1',
      targetId: 'android-arm64-vulkan',
      profile: 'android-debug',
    },
  ];
  return { root, file, manifest, artifacts };
};

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('Build artifact reveal authorization', () => {
  it('allows only recorded artifact and manifest paths in the active project', () => {
    const { root, file, manifest, artifacts } = fixture();
    expect(resolveBuildArtifactPath(root, artifacts, artifacts[0].path)).toBe(fs.realpathSync(file));
    expect(resolveBuildArtifactPath(root, artifacts, artifacts[0].manifestPath)).toBe(fs.realpathSync(manifest));
    expect(() => resolveBuildArtifactPath(root, artifacts, 'Build/Packages/secrets.txt')).toThrow(
      'Artifact does not belong',
    );
    expect(() => resolveBuildArtifactPath(root, [], artifacts[0].path)).toThrow('Artifact does not belong');
  });

  it('rejects traversal and paths that leave the active project', () => {
    const { root, artifacts } = fixture();
    const parent = path.dirname(root);
    expect(() => resolveBuildArtifactPath(root, [{ ...artifacts[0], path: '../other.txt' }], '../other.txt')).toThrow(
      'escapes the active project',
    );
    expect(() => resolveBuildArtifactPath(root, [{ ...artifacts[0], path: parent }], parent)).toThrow(
      'escapes the active project',
    );
  });

  it('rejects a recorded symlink that resolves outside the project', () => {
    if (process.platform === 'win32') return;
    const { root, artifacts } = fixture();
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-artifact-outside-'));
    roots.push(outside);
    fs.writeFileSync(path.join(outside, 'other.apk'), 'unrelated');
    const link = path.join(root, 'Build', 'Packages', 'linked.apk');
    fs.symlinkSync(path.join(outside, 'other.apk'), link);
    expect(() =>
      resolveBuildArtifactPath(
        root,
        [{ ...artifacts[0], path: 'Build/Packages/linked.apk' }],
        'Build/Packages/linked.apk',
      ),
    ).toThrow('resolves outside the active project');
  });
});
