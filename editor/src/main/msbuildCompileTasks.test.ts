import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { ArcProjectCandidate } from '../common/projectTypes';
import { beginMsbuildCompileTrace, parseMsbuildCompileLog } from './msbuildCompileTasks';

const roots: string[] = [];
function fixture(generator?: string) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-msbuild-timing-'));
  roots.push(root);
  const build = path.join(root, 'Build', 'default');
  fs.mkdirSync(build, { recursive: true });
  if (generator) fs.writeFileSync(path.join(build, 'CMakeCache.txt'), 'CMAKE_GENERATOR:INTERNAL=' + generator + '\n');
  return {
    build,
    project: { projectRoot: root, descriptor: { paths: { build: 'Build' } } } as ArcProjectCandidate,
  };
}
const runId = 'eb97d024-98ac-49b4-8791-1ab76462a4a7';
const log = [
  '21:15:02.100 1>  Task "CL" (TaskId:17)',
  '21:15:02.125 2>  Task "CL" (TaskId:22)',
  '21:15:02.175 2>  Done executing task "CL". (TaskId:22)',
  '21:15:02.250 1>  Done executing task "CL". (TaskId:17)',
  '',
].join('\n');

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('BM7.2 native MSBuild CL task intervals', () => {
  it('pairs concurrent native CL start/finish events and preserves their overlap', () => {
    const result = parseMsbuildCompileLog(log);
    expect(result).toEqual({
      source: 'msbuild-file-log',
      clock: 'msbuild-cl-events',
      truncated: false,
      tasks: [
        { output: 'CL task #17 (node 1)', startMs: 0, endMs: 150, durationMs: 150 },
        { output: 'CL task #22 (node 2)', startMs: 25, endMs: 75, durationMs: 50 },
      ],
    });
  });

  it('does not invent timings for no-op builds, unmatched tasks, or non-CL tools', () => {
    expect(parseMsbuildCompileLog('21:15:02.100 1>  Task "Link" (TaskId:17)\n')).toBeUndefined();
    expect(parseMsbuildCompileLog('21:15:02.100 1>  Done executing task "CL". (TaskId:7)\n')).toBeUndefined();
    expect(parseMsbuildCompileLog('1>  Task "CL" (TaskId:4)\n')).toBeUndefined();
    expect(parseMsbuildCompileLog('21:15:02.100 1>  Task "CL" (TaskId:7)\n')).toEqual({
      source: 'msbuild-file-log',
      clock: 'msbuild-cl-events',
      tasks: [],
      truncated: true,
    });
  });

  it('normalizes midnight rollover without confusing historical builds', () => {
    const input = [
      '23:59:59.950 1>Task "CL" (TaskId:13)',
      '00:00:00.150 1>Done executing task "CL". (TaskId:13)',
      '',
    ].join('\n');
    expect(parseMsbuildCompileLog(input)?.tasks).toEqual([
      { output: 'CL task #13 (node 1)', startMs: 0, endMs: 200, durationMs: 200 },
    ]);
  });

  it('uses a fresh, build-local log per invocation and deletes raw diagnostics after parsing', () => {
    const { project } = fixture('Visual Studio 17 2022');
    const capture = beginMsbuildCompileTrace(project, runId, 'win32');
    expect(capture).not.toBeNull();
    expect(capture?.args).toEqual([
      '--msbuild-timing-log',
      path.join(project.projectRoot, 'Build', 'default', '.arc-monitor', 'arc-msbuild-' + runId + '.log'),
    ]);
    const filename = capture!.args[1];
    fs.writeFileSync(filename, log);
    expect(capture?.finish()?.tasks).toHaveLength(2);
    expect(fs.existsSync(filename)).toBe(false);
    expect(capture?.finish()).toBeUndefined();
  });

  it('reads only completed CL intervals during a running build without deleting the log', () => {
    const { project } = fixture('Visual Studio 17 2022');
    const capture = beginMsbuildCompileTrace(project, runId, 'win32');
    expect(capture).not.toBeNull();
    const filename = capture!.args[1];
    expect(capture?.snapshot()).toBeUndefined();

    const starts = '12:00:00.000 1>  Task "CL" (TaskId:9)\n' + '12:00:00.100 2>  Task "CL" (TaskId:10)\n';
    fs.writeFileSync(filename, starts);
    expect(capture?.snapshot()).toMatchObject({ tasks: [], truncated: false });
    fs.appendFileSync(filename, '12:00:00.320 2>  Done executing task "CL". (TaskId:10)\n');
    expect(capture?.snapshot()).toMatchObject({
      tasks: [{ output: 'CL task #10 (node 2)', startMs: 0, endMs: 220, durationMs: 220 }],
      truncated: false,
    });
    expect(fs.existsSync(filename)).toBe(true);
    fs.appendFileSync(filename, '12:00:00.600 1>  Done executing task "CL". (TaskId:9)\n');
    expect(capture?.finish()?.tasks).toHaveLength(2);
    expect(fs.existsSync(filename)).toBe(false);
  });

  it('associates only source inputs logged by the matching CL task, without per-file durations', () => {
    const { project, build } = fixture('Visual Studio 17 2022');
    const input = [
      '21:15:02.100 1> Task "CL" (TaskId:17)',
      '21:15:02.101 1> Task Parameter:Sources=',
      '21:15:02.102 1> ../../Source/first.cpp',
      '21:15:02.103 2> Task "CL" (TaskId:22)',
      '21:15:02.104 2> Task Parameter:Sources=',
      '21:15:02.105 2> ../../Source/other.cpp',
      '21:15:02.106 1> ../../Source/second.cpp',
      '21:15:02.175 2> Done executing task "CL". (TaskId:22)',
      '21:15:02.250 1> Done executing task "CL". (TaskId:17)',
      '',
    ].join('\n');
    const value = parseMsbuildCompileLog(input, false, true, project.projectRoot, build);
    expect(value?.tasks).toEqual([
      {
        output: 'CL task #17 (node 1)',
        sourceFiles: ['Source/first.cpp', 'Source/second.cpp'],
        startMs: 0,
        endMs: 150,
        durationMs: 150,
      },
      {
        output: 'CL task #22 (node 2)',
        sourceFiles: ['Source/other.cpp'],
        sourceFile: 'Source/other.cpp',
        startMs: 3,
        endMs: 75,
        durationMs: 72,
      },
    ]);
    expect(value?.activeTasks).toBeUndefined();
  });

  it('publishes in-flight native CL tasks without invented duration and removes them at completion', () => {
    const { project, build } = fixture('Visual Studio 17 2022');
    const lines = [
      '21:15:02.100 1> Task "CL" (TaskId:17)',
      '21:15:02.101 1> Task Parameter:Sources=../../Source/main.cpp',
      '21:15:02.125 2> Task "CL" (TaskId:22)',
      '',
    ];
    const current = parseMsbuildCompileLog(lines.join('\n'), false, false, project.projectRoot, build);
    expect(current).toMatchObject({
      tasks: [],
      truncated: false,
      activeTasks: [
        { output: 'CL task #17 (node 1)', sourceFiles: ['Source/main.cpp'] },
        { output: 'CL task #22 (node 2)' },
      ],
    });
    expect(JSON.stringify(current)).not.toContain('durationMs');
    const finished = parseMsbuildCompileLog(
      [...lines.slice(0, -1), '21:15:02.400 1> Done executing task "CL". (TaskId:17)', ''].join('\n'),
      false,
      false,
      project.projectRoot,
      build,
    );
    expect(finished?.tasks[0]).toMatchObject({
      output: 'CL task #17 (node 1)',
      sourceFiles: ['Source/main.cpp'],
      durationMs: 300,
    });
    expect(finished?.activeTasks).toEqual([{ output: 'CL task #22 (node 2)' }]);
    const final = parseMsbuildCompileLog(lines.join('\n'), false, true, project.projectRoot, build);
    expect(final?.activeTasks).toBeUndefined();
  });

  it('does not claim source ownership for ambiguous nodes or paths outside the project', () => {
    const { project, build } = fixture('Visual Studio 17 2022');
    const input = [
      '10:00:00.000 1> Task "CL" (TaskId:17)',
      '10:00:00.010 1> Task "CL" (TaskId:18)',
      '10:00:00.020 1> Task Parameter:Sources=../../Source/ambiguous.cpp',
      '10:00:00.030 2> Task Parameter:Sources=../../Source/wrong-node.cpp',
      '10:00:00.040 1> Task Parameter:Sources=../../../../outside.cpp',
      '10:00:00.050 1> Done executing task "CL". (TaskId:17)',
      '10:00:00.060 1> Done executing task "CL". (TaskId:18)',
      '',
    ].join('\n');
    const value = parseMsbuildCompileLog(input, false, true, project.projectRoot, build);
    expect(value?.tasks).toHaveLength(2);
    expect(value?.tasks.every((task) => task.sourceFiles === undefined)).toBe(true);
  });

  it('does not attach MSBuild flags for Ninja, non-Windows hosts, or invalid run identities', () => {
    const { project } = fixture('Ninja');
    expect(beginMsbuildCompileTrace(project, runId, 'win32')).toBeNull();
    expect(beginMsbuildCompileTrace(project, runId, 'linux')).toBeNull();
    expect(beginMsbuildCompileTrace(project, '../unsafe', 'win32')).toBeNull();
  });

  it('rejects redirected diagnostic directories and symlinked output files', () => {
    if (process.platform === 'win32') return;
    const { project, build } = fixture('Visual Studio 17 2022');
    const folder = path.join(build, '.arc-monitor');
    const outside = path.join(project.projectRoot, 'outside');
    fs.mkdirSync(outside);
    fs.symlinkSync(outside, folder);
    expect(beginMsbuildCompileTrace(project, runId, 'win32')).toBeNull();
    fs.rmSync(folder);
    const capture = beginMsbuildCompileTrace(project, runId, 'win32');
    expect(capture).not.toBeNull();
    const target = path.join(outside, 'existing.txt');
    fs.writeFileSync(target, 'external');
    fs.symlinkSync(target, capture!.args[1]);
    expect(capture?.finish()).toBeUndefined();
    expect(fs.readFileSync(target, 'utf8')).toBe('external');
  });
});
