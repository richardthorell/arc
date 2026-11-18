import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { ArcProjectCandidate } from '../common/projectTypes';
import { beginNinjaCompileTrace } from './ninjaCompileTasks';

const roots: string[] = [];
function fixture() {
  const projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-ninja-trace-'));
  roots.push(projectRoot);
  const build = path.join(projectRoot, 'Build', 'default');
  const filename = path.join(build, '.ninja_log');
  const project = {
    projectRoot,
    descriptor: { paths: { build: 'Build' } },
  } as ArcProjectCandidate;
  return { project, build, filename };
}
const header = '# ninja log v5\n';
const record = (start: number, end: number, output: string) =>
  [start, end, 123456789, output, 'abcd1234'].join('\t') + '\n';

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('BM7.1 Ninja measured compilation tasks', () => {
  it('collects parallel object compilations from a first build, excluding non-compiler edges', () => {
    const { project, build, filename } = fixture();
    const finish = beginNinjaCompileTrace(project);
    expect(finish).not.toBeNull();
    fs.mkdirSync(build, { recursive: true });
    fs.writeFileSync(
      filename,
      header +
        record(12, 125, 'CMakeFiles/game.dir/main.cpp.o') +
        record(15, 75, 'CMakeFiles/game.dir/math.cpp.o') +
        record(125, 200, 'game.exe'),
    );
    expect(finish?.()).toEqual({
      source: 'ninja-log',
      clock: 'ninja-invocation',
      truncated: false,
      tasks: [
        { output: 'CMakeFiles/game.dir/main.cpp.o', startMs: 12, endMs: 125, durationMs: 113 },
        { output: 'CMakeFiles/game.dir/math.cpp.o', startMs: 15, endMs: 75, durationMs: 60 },
      ],
    });
  });

  it('does not inherit compile timings from older Ninja invocations or a no-op build', () => {
    const { project, build, filename } = fixture();
    fs.mkdirSync(build, { recursive: true });
    fs.writeFileSync(filename, header + record(0, 10000, 'CMakeFiles/game.dir/old.cpp.obj'));
    const first = beginNinjaCompileTrace(project);
    expect(first?.()).toBeUndefined();
    fs.appendFileSync(filename, record(10, 55, 'CMakeFiles/game.dir/new.cpp.obj'));
    expect(first?.()?.tasks).toEqual([
      { output: 'CMakeFiles/game.dir/new.cpp.obj', startMs: 10, endMs: 55, durationMs: 45 },
    ]);
  });

  it('ignores stale log files when a failed configuration never started Ninja', () => {
    const { project, build, filename } = fixture();
    fs.mkdirSync(build, { recursive: true });
    fs.writeFileSync(filename, header + record(0, 10, 'old.obj'));
    const finish = beginNinjaCompileTrace(project);
    expect(finish?.()).toBeUndefined();
  });

  it('rejects unsupported versions, backwards timestamps and a truncated log', () => {
    const { project, build, filename } = fixture();
    fs.mkdirSync(build, { recursive: true });
    fs.writeFileSync(filename, '# ninja log v4\n');
    let finish = beginNinjaCompileTrace(project);
    fs.appendFileSync(filename, record(1, 2, 'bad.obj'));
    expect(finish?.()).toBeUndefined();

    fs.writeFileSync(filename, header);
    finish = beginNinjaCompileTrace(project);
    fs.appendFileSync(filename, record(10, 1, 'bad.obj') + record(1, 25, 'good.obj'));
    expect(finish?.()?.tasks).toEqual([{ output: 'good.obj', startMs: 1, endMs: 25, durationMs: 24 }]);

    const cursor = beginNinjaCompileTrace(project);
    fs.writeFileSync(filename, header);
    expect(cursor?.()).toBeUndefined();
  });

  it('does not follow symlinked Ninja logs', () => {
    if (process.platform === 'win32') return;
    const { project, build, filename } = fixture();
    fs.mkdirSync(build, { recursive: true });
    const outside = path.join(build, 'outside.txt');
    fs.writeFileSync(outside, header);
    fs.symlinkSync(outside, filename);
    expect(beginNinjaCompileTrace(project)).toBeNull();
  });
});
