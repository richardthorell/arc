import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { ArcProjectCandidate } from '../common/projectTypes';
import type { BuildCompileTrace } from '../common/buildCompileTypes';
import type { BuildTargetGraph } from '../common/buildTargetTypes';
import { enrichCompileTrace } from './compileTaskAttribution';

const roots: string[] = [];
function writeDatabase(filename: string, entries: unknown[]): void {
  fs.writeFileSync(filename, JSON.stringify(entries));
}
function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-source-association-'));
  roots.push(root);
  const build = path.join(root, 'Build', 'default');
  fs.mkdirSync(build, { recursive: true });
  const project = {
    projectRoot: root,
    descriptor: { paths: { build: 'Build' } },
  } as ArcProjectCandidate;
  const filename = path.join(build, 'compile_commands.json');
  const entry = (file: string, output: string) => ({
    directory: build,
    file: path.join(root, file),
    command: 'c++ -c ' + path.join(root, file) + ' -o ' + output,
  });
  const trace: BuildCompileTrace = {
    source: 'ninja-log',
    clock: 'ninja-invocation',
    truncated: false,
    tasks: [
      { output: 'CMakeFiles/MyEditor.dir/Source/main.cpp.o', startMs: 10, endMs: 45, durationMs: 35 },
      { output: 'CMakeFiles/Other.dir/Source/other.cpp.o', startMs: 8, endMs: 51, durationMs: 43 },
      { output: 'CMakeFiles/MyEditor.dir/Source/unknown.cpp.o', startMs: 2, endMs: 67, durationMs: 65 },
    ],
  };
  const graph: BuildTargetGraph = {
    revision: 1,
    generator: 'Ninja',
    configuration: 'Debug',
    discoveredAt: '2026-11-18T10:00:00Z',
    rootTargetId: 'editor',
    truncated: false,
    nodes: [{ id: 'editor', name: 'MyEditor', type: 'MODULE_LIBRARY', dependencies: [], artifacts: [] }],
  };
  return { root, build, project, filename, entry, trace, graph };
}

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('BM7.3 verified native source-file association', () => {
  it('joins exact object paths, preserves native time intervals and verifies target names against BM6', () => {
    const { filename, entry, project, graph, trace } = fixture();
    writeDatabase(filename, [
      entry('Source/main.cpp', 'CMakeFiles/MyEditor.dir/Source/main.cpp.o'),
      entry('Source/other.cpp', 'CMakeFiles/Other.dir/Source/other.cpp.o'),
    ]);
    const value = enrichCompileTrace(project, trace, graph);
    expect(value.tasks).toEqual([
      { ...trace.tasks[0], sourceFile: 'Source/main.cpp', targetName: 'MyEditor' },
      { ...trace.tasks[1], sourceFile: 'Source/other.cpp' },
      trace.tasks[2],
    ]);
    expect(value.clock).toBe('ninja-invocation');
  });

  it('accepts explicit object outputs and argument-array outputs but never matches source basenames alone', () => {
    const { filename, project, trace, build, root } = fixture();
    writeDatabase(filename, [
      {
        directory: build,
        file: path.join(root, 'Source', 'main.cpp'),
        output: path.join(build, 'CMakeFiles/MyEditor.dir/Source/main.cpp.o'),
      },
      {
        directory: build,
        file: path.join(root, 'Source', 'other.cpp'),
        arguments: ['c++', '-c', 'Source/other.cpp', '-o', 'CMakeFiles/Other.dir/Source/other.cpp.o'],
      },
      {
        directory: build,
        file: path.join(root, 'Source', 'unknown.cpp'),
        command: 'c++ -c unknown.cpp',
      },
    ]);
    const value = enrichCompileTrace(project, trace);
    expect(value.tasks[0].sourceFile).toBe('Source/main.cpp');
    expect(value.tasks[1].sourceFile).toBe('Source/other.cpp');
    expect(value.tasks[0].targetName).toBeUndefined();
    expect(value.tasks[2].sourceFile).toBeUndefined();
  });

  it('refuses conflicting source identities for the same object', () => {
    const { filename, entry, project, trace } = fixture();
    writeDatabase(filename, [
      entry('Source/main.cpp', trace.tasks[0].output),
      entry('Source/different.cpp', trace.tasks[0].output),
    ]);
    expect(enrichCompileTrace(project, trace).tasks[0]).toEqual(trace.tasks[0]);
  });

  it('does not expose source files outside the project or object outputs outside the build directory', () => {
    const { filename, entry, project, trace } = fixture();
    writeDatabase(filename, [
      entry('../../secrets.cpp', trace.tasks[0].output),
      entry('Source/other.cpp', '../outside.o'),
    ]);
    expect(enrichCompileTrace(project, trace)).toEqual(trace);
  });

  it('ignores missing and symlinked databases and leaves MSBuild tasks unmodified', () => {
    const { filename, entry, project, trace, graph, build } = fixture();
    expect(enrichCompileTrace(project, trace, graph)).toEqual(trace);
    const msbuild: BuildCompileTrace = {
      source: 'msbuild-file-log',
      clock: 'msbuild-cl-events',
      truncated: false,
      tasks: [{ output: 'CL task #1', startMs: 0, endMs: 10, durationMs: 10 }],
    };
    writeDatabase(filename, [entry('Source/main.cpp', trace.tasks[0].output)]);
    expect(enrichCompileTrace(project, msbuild, graph)).toEqual(msbuild);
    if (process.platform === 'win32') return;
    fs.rmSync(filename);
    const outside = path.join(build, 'outside.json');
    fs.writeFileSync(outside, JSON.stringify([entry('Source/main.cpp', trace.tasks[0].output)]));
    fs.symlinkSync(outside, filename);
    expect(enrichCompileTrace(project, trace, graph)).toEqual(trace);
  });
});
