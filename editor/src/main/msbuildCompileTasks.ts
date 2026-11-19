import fs from 'node:fs';
import path from 'node:path';

import type { BuildCompileTask, BuildCompileTrace } from '../common/buildCompileTypes';
import type { ArcProjectCandidate } from '../common/projectTypes';
import { editorCommandBuildDirectory } from './editorCommandTargetGraph';

const MAX_LOG_BYTES = 16 * 1024 * 1024;
const MAX_TASKS = 256;
const MAX_TRACE_BYTES = 24 * 1024;
const DAY_MS = 24 * 60 * 60 * 1000;
const MSBUILD_LOG_DIR = '.arc-monitor';

/** Native MSBuild file logger timestamps, in the form HH:mm:ss[.fffffff]. */
function clockMs(line: string): number | null {
  const match = /^\s*(?:\d{4}-\d{2}-\d{2}[ T])?(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,7}))?\s/.exec(line);
  if (!match) return null;
  const h = Number(match[1]);
  const m = Number(match[2]);
  const s = Number(match[3]);
  if (h >= 24 || m >= 60 || s >= 60) return null;
  const milliseconds = Number((match[4] ?? '').padEnd(3, '0').slice(0, 3));
  return ((h * 60 + m) * 60 + s) * 1000 + milliseconds;
}

/** Parses MSBuild's real TaskStarted/TaskFinished log messages, not console receive times.
 * A single CL task can compile multiple sources, so these are measured CL
 * invocation intervals and MUST NOT be labelled per-source/object timings.
 */
export function parseMsbuildCompileLog(
  content: string,
  clipped = false,
  completed = true,
): BuildCompileTrace | undefined {
  const active = new Map<string, { start: number; label: string }>();
  const intervals: Array<{ start: number; end: number; label: string }> = [];
  const startPattern = /\bTask "CL"\s*\(TaskId:\s*(\d+)\)/;
  const endPattern = /\bDone executing task "CL"\.\s*\(TaskId:\s*(\d+)\)/;
  let day = 0;
  let lastClock: number | undefined;
  let observed = false;
  for (const line of content.split(/\r?\n/)) {
    const start = startPattern.exec(line);
    const end = endPattern.exec(line);
    if (!start && !end) continue;
    const time = clockMs(line);
    if (time === null) continue;
    if (lastClock !== undefined && lastClock - time > DAY_MS / 2) day++;
    lastClock = time;
    const at = day * DAY_MS + time;
    const prefix = line.slice(0, (start ?? end)!.index);
    const node = /\b(\d+)>\s*/.exec(prefix)?.[1] ?? '';
    const taskId = (start ?? end)![1];
    const key = node + ':' + taskId;
    if (start) {
      observed = true;
      if (!active.has(key)) {
        active.set(key, { start: at, label: 'CL task #' + taskId + (node ? ' (node ' + node + ')' : '') });
      }
    } else {
      const task = active.get(key);
      if (!task || at < task.start) continue;
      active.delete(key);
      intervals.push({ start: task.start, end: at, label: task.label });
    }
  }
  if (!observed) return undefined;
  const base = Math.min(...intervals.map((task) => task.start));
  const tasks: BuildCompileTask[] = [];
  let bytes = 0;
  let truncated = clipped;
  for (const task of intervals) {
    if (!Number.isFinite(base)) break;
    const size = Buffer.byteLength(task.label, 'utf8') + 96;
    if (tasks.length >= MAX_TASKS || bytes + size > MAX_TRACE_BYTES) {
      truncated = true;
      break;
    }
    bytes += size;
    tasks.push({
      output: task.label,
      startMs: task.start - base,
      endMs: task.end - base,
      durationMs: task.end - task.start,
    });
  }
  tasks.sort((a, b) => a.startMs - b.startMs);
  // A failed/cancelled build may have incomplete tasks; never fabricate an end time.
  return {
    source: 'msbuild-file-log',
    clock: 'msbuild-cl-events',
    tasks,
    truncated: truncated || (completed && active.size > 0),
  };
}

export type MsbuildCompileCapture = {
  /** Pass only to the native arc-project CLI, not directly to CMake. */
  args: string[];
  /** Read completed CL intervals without deleting the in-progress native log. */
  snapshot: () => BuildCompileTrace | undefined;
  finish: () => BuildCompileTrace | undefined;
};

/**
 * The editor opts into the native CLI's MSBuild file logger with a unique
 * project-local path per build. Only MSBuild/Visual Studio generates this log;
 * Ninja and other generators continue using BM7.1 unchanged.
 */
export function beginMsbuildCompileTrace(
  project: ArcProjectCandidate,
  runId: string,
  host = process.platform,
): MsbuildCompileCapture | null {
  if (host !== 'win32' || !/^[0-9a-f-]{36}$/i.test(runId)) return null;
  const build = editorCommandBuildDirectory(project);
  if (!build) return null;
  try {
    const cache = path.join(build, 'CMakeCache.txt');
    if (fs.existsSync(cache)) {
      const generator = fs.readFileSync(cache, 'utf8').match(/^CMAKE_GENERATOR:INTERNAL=(.+)$/m)?.[1];
      if (generator && !generator.startsWith('Visual Studio ')) return null;
    }
    const folder = path.join(build, MSBUILD_LOG_DIR);
    fs.mkdirSync(folder, { recursive: true });
    const stat = fs.lstatSync(folder);
    if (stat.isSymbolicLink() || !stat.isDirectory()) return null;
    const filename = path.join(folder, 'arc-msbuild-' + runId + '.log');
    if (fs.existsSync(filename) || filename.includes(';')) return null;
    const readTrace = (completed: boolean): BuildCompileTrace | undefined => {
      try {
        const stat = fs.lstatSync(filename);
        if (stat.isSymbolicLink() || !stat.isFile()) return undefined;
        const fd = fs.openSync(filename, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0));
        try {
          const size = Math.min(stat.size, MAX_LOG_BYTES);
          const buffer = Buffer.alloc(size);
          const read = fs.readSync(fd, buffer, 0, size, 0);
          const end = buffer.lastIndexOf(10, read - 1);
          if (end < 0) return undefined;
          return parseMsbuildCompileLog(buffer.subarray(0, end).toString('utf8'), stat.size > MAX_LOG_BYTES, completed);
        } finally {
          fs.closeSync(fd);
        }
      } catch {
        // An active MSBuild logger may temporarily lock or buffer the file.
        // Unavailable telemetry must never interfere with compilation.
        return undefined;
      }
    };
    let lastSize = -1;
    let lastMtime = -1;
    let lastTrace: BuildCompileTrace | undefined;
    return {
      args: ['--msbuild-timing-log', filename],
      snapshot: () => {
        try {
          const info = fs.lstatSync(filename);
          if (!info.isFile() || info.isSymbolicLink()) return undefined;
          if (info.size === lastSize && info.mtimeMs === lastMtime) return lastTrace;
          const next = readTrace(false);
          // A locked logger can return nothing even if the file exists.
          // Retry on the next poll rather than caching unavailable data.
          if (next) {
            lastTrace = next;
            lastSize = info.size;
            lastMtime = info.mtimeMs;
          }
          return next;
        } catch {
          return undefined;
        }
      },
      finish: () => {
        try {
          return readTrace(true);
        } finally {
          try {
            // Persist bounded intervals, not raw potentially sensitive logs.
            fs.rmSync(filename, { force: true });
          } catch {
            // Cleanup failures must not change the native command's result.
          }
        }
      },
    };
  } catch {
    return null;
  }
}
