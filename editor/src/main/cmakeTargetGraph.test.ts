import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vitest';
import { CMakeTargetOutputObserver, readCMakeTargetGraph, requestCMakeCodemodel } from './cmakeTargetGraph';

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

function fixture() {
  const buildDir = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-file-api-'));
  roots.push(buildDir);
  requestCMakeCodemodel(buildDir);
  const reply = path.join(buildDir, '.cmake', 'api', 'v1', 'reply');
  fs.mkdirSync(reply, { recursive: true });
  fs.writeFileSync(path.join(reply, 'index-20261116-100000.json'), JSON.stringify({
    cmake: { generator: { name: 'Visual Studio 17 2022' } },
    reply: { 'codemodel-v2': { jsonFile: 'codemodel-v2-file.json' } },
  }));
  fs.writeFileSync(path.join(reply, 'codemodel-v2-file.json'), JSON.stringify({
    configurations: [{
      name: 'Shipping',
      targets: [
        { id: 'root', name: 'arc-runtime-output', jsonFile: 'target-root.json' },
        { id: 'game', name: 'game', jsonFile: 'target-game.json' },
        { id: 'core', name: 'arc_core', jsonFile: 'target-core.json' },
        { id: 'unrelated', name: 'test_runner', jsonFile: 'target-test.json' },
      ],
    }],
  }));
  const node = (id: string, name: string, type: string, dependencies: string[], artifacts: string[]) =>
    JSON.stringify({ id, name, type, dependencies: dependencies.map((dependency) => ({ id: dependency })),
      artifacts: artifacts.map((artifact) => ({ path: artifact })) });
  fs.writeFileSync(path.join(reply, 'target-root.json'), node('root', 'arc-runtime-output', 'UTILITY', ['game'], []));
  fs.writeFileSync(path.join(reply, 'target-game.json'), node('game', 'game', 'EXECUTABLE', ['core'], ['Shipping/game.exe']));
  fs.writeFileSync(path.join(reply, 'target-core.json'), node('core', 'arc_core', 'STATIC_LIBRARY', [], ['Shipping/arc_core.lib']));
  fs.writeFileSync(path.join(reply, 'target-test.json'), node('unrelated', 'test_runner', 'EXECUTABLE', [], ['Shipping/test.exe']));
  return { buildDir, reply };
}

describe('CMake File API target graph', () => {
  it('writes a codemodel query and selects only reachable native dependencies', () => {
    const { buildDir } = fixture();
    expect(fs.existsSync(path.join(buildDir, '.cmake', 'api', 'v1', 'query', 'codemodel-v2'))).toBe(true);
    const graph = readCMakeTargetGraph(buildDir, 'Shipping', 'arc-runtime-output');
    expect(graph).toMatchObject({ revision: 1, configuration: 'Shipping', rootTargetId: 'root', truncated: false });
    expect(graph?.nodes.map((node) => [node.id, node.name, node.dependencies])).toEqual([
      ['root', 'arc-runtime-output', ['game']],
      ['game', 'game', ['core']],
      ['core', 'arc_core', []],
    ]);
    expect(graph?.nodes.find((item) => item.name === 'test_runner')).toBeUndefined();
    expect(readCMakeTargetGraph(buildDir, 'Debug', 'arc-runtime-output')).toBeNull();
    expect(readCMakeTargetGraph(buildDir, 'Shipping', 'missing')).toBeNull();
  });

  it('observes only exact File API artifacts, once, across stdout chunk boundaries', () => {
    const { buildDir } = fixture();
    const graph = readCMakeTargetGraph(buildDir, 'Shipping', 'arc-runtime-output')!;
    const seen = vi.fn();
    const observer = new CMakeTargetOutputObserver(buildDir, graph, seen);
    const artifact = path.join(buildDir, 'Shipping', 'game.exe');
    observer.accept('  Game.vcxproj -> ', 'stdout');
    expect(seen).not.toHaveBeenCalled();
    observer.accept(`${artifact}\n`, 'stdout');
    observer.accept(`  Game.vcxproj -> ${artifact}\n`, 'stdout');
    observer.accept(`  Unknown.vcxproj -> ${path.join(buildDir, 'Shipping', 'not-in-graph.exe')}\n`, 'stdout');
    observer.accept('Progress 5/10 compiled\n', 'stdout');
    expect(seen).toHaveBeenCalledTimes(1);
    expect(seen).toHaveBeenCalledWith('game');
    // An incremental build can emit no artifacts; it must not invent progress.
    const clean = vi.fn();
    new CMakeTargetOutputObserver(buildDir, graph, clean).accept('Build up-to-date\n', 'stdout');
    expect(clean).not.toHaveBeenCalled();
  });

  it('rejects traversal in CMake-provided reply filenames', () => {
    const { buildDir, reply } = fixture();
    fs.writeFileSync(path.join(reply, 'index-20261116-200000.json'), JSON.stringify({
      reply: { 'codemodel-v2': { jsonFile: '../outside.json' } },
    }));
    expect(() => readCMakeTargetGraph(buildDir, 'Shipping', 'arc-runtime-output'))
      .toThrow('Invalid CMake File API filename');
  });

  it('does not invent a graph if the native reply is unavailable', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-file-api-empty-'));
    roots.push(root);
    requestCMakeCodemodel(root);
    expect(readCMakeTargetGraph(root, 'Shipping', 'arc-runtime-output')).toBeNull();
  });
});
