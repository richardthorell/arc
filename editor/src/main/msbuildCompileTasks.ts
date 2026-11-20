import fs from 'node:fs';
import path from 'node:path';

import type { BuildCompileTask, BuildCompileTrace } from '../common/buildCompileTypes';
import type { ArcProjectCandidate } from '../common/projectTypes';
import { editorCommandBuildDirectory } from './editorCommandTargetGraph';

const MAX_LOG_BYTES = 16 * 1024 * 1024;
const MAX_TASKS = 256;
const MAX_TRACE_BYTES = 24 * 1024;
const MAX_ACTIVE_TASKS = 24;
const MAX_SOURCE_FILES = 8;
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


type ClEvent = {
  start: number;
  label: string;
  sources: string[];
  sourceFilesTruncated: boolean;
};

function safeSource(raw: string, projectRoot?: string, buildRoot?: string): string | null {
  if (!projectRoot || !buildRoot) return null;
  const value = raw.trim().replace(/^["']|["']$/g, '');
  if (value.length > 1024 || /[\0\r\n;]/u.test(value) || !/\.(?:c|cc|cpp|cxx|ixx)$/iu.test(value)) return null;
  const absolute = path.resolve(buildRoot, value);
  const relative = path.relative(path.resolve(projectRoot), absolute);
  if (!relative || relative.length > 240 || relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative)) return null;
  return relative.replace(/\\/g, '/');
}

/** The multi-process file logger prefixes messages with timestamps and/or node IDs. */
function logMessage(line: string): { message: string; node: string } {
  const withoutTime = line.replace(/^\s*(?:\d{4}-\d{2}-\d{2}[ T])?\d{1,2}:\d{2}:\d{2}(?:\.\d{1,7})?\s+/u, '');
  const match = /^\s*(\d+)>\s*/u.exec(withoutTime);
  return { message: (match ? withoutTime.slice(match[0].length) : withoutTime).trim(), node: match?.[1] ?? '' };
}

function taskSources(task: ClEvent): { sourceFiles?: string[]; sourceFilesTruncated?: boolean; sourceFile?: string } {
  const sources = task.sources;
  return {
    ...(sources.length ? { sourceFiles: sources } : {}),
    ...(task.sourceFilesTruncated ? { sourceFilesTruncated: true } : {}),
    ...(sources.length === 1 && !task.sourceFilesTruncated ? { sourceFile: sources[0] } : {}),
  };
}

/** Parses MSBuild's real TaskStarted/TaskFinished log messages, not console receive times.
 * A single CL task can compile multiple sources, so these are measured CL
 * invocation intervals and MUST NOT be labelled per-source/object timings.
 */
export function parseMsbuildCompileLog(
  content: string,
  clipped = false,
  completed = true,
  projectRoot?: string,
  buildRoot?: string,
): BuildCompileTrace | undefined {
  const active = new Map<string, ClEvent>();
  const intervals: Array<{ start: number; end: number; task: ClEvent }> = [];
  const startPattern = /\bTask "CL"\s*\(TaskId:\s*(\d+)\)/;
  const endPattern = /\bDone executing task "CL"\.\s*\(TaskId:\s*(\d+)\)/;
  let day = 0;
  let lastClock: number | undefined;
  let observed = false;
  let sourceOwner: string | null = null;
  for (const line of content.split(/\r?\n/)) {
    const start = startPattern.exec(line);
    const end = endPattern.exec(line);
    if (!start && !end) {
      const { message, node } = logMessage(line);
      const parameter = /\bTask Parameter:([A-Za-z]+)=\s*(.*)$/iu.exec(message);
      if (parameter) {
        sourceOwner = null;
        if (parameter[1].toLowerCase() !== 'sources') continue;
        // When multiple CL tasks run on the same node, an untagged Task
        // Parameter line cannot be associated reliably with a specific task.
        const candidates = [...active.entries()].filter(([key]) => !node || key.startsWith(node + ':'));
        if (candidates.length !== 1) continue;
        sourceOwner = candidates[0][0];
        const value = safeSource(parameter[2], projectRoot, buildRoot);
        if (value) {
          const current = active.get(sourceOwner);
          if (current && !current.sources.includes(value)) current.sources.push(value);
        }
        continue;
      }
      if (/\b(?:Task |Done executing task |Output Item\(s\):|Added Item\(s\):)/iu.test(message)) {
        sourceOwner = null;
        continue;
      }
      const task = sourceOwner ? active.get(sourceOwner) : undefined;
      if (task && (!node || sourceOwner!.startsWith(node + ':'))) {
        const value = safeSource(message, projectRoot, buildRoot);
        if (value && !task.sources.includes(value)) {
          if (task.sources.length < MAX_SOURCE_FILES) task.sources.push(value);
          else task.sourceFilesTruncated = true;
        } else if (message && !value) {
          sourceOwner = null;
        }
      }
      continue;
    }
    sourceOwner = null;
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
        active.set(key, {
          start: at,
          label: 'CL task #' + taskId + (node ? ' (node ' + node + ')' : ''),
          sources: [],
          sourceFilesTruncated: false,
        });
      }
    } else {
      const task = active.get(key);
      if (!task || at < task.start) continue;
      active.delete(key);
      intervals.push({ start: task.start, end: at, task });
    }
  }
  if (!observed) return undefined;
  const base = Math.min(...intervals.map((task) => task.start));
  const tasks: BuildCompileTask[] = [];
  let bytes = 0;
  let truncated = clipped;
  for (const task of intervals) {
    if (!Number.isFinite(base)) break;
    const sourceFields = taskSources(task.task);
    const size = Buffer.byteLength(task.task.label, 'utf8') + Buffer.byteLength(JSON.stringify(sourceFields), 'utf8') + 96;
    if (tasks.length >= MAX_TASKS || bytes + size > MAX_TRACE_BYTES) {
      truncated = true;
      break;
    }
    bytes += size;
    tasks.push({
      output: task.task.label,
      ...sourceFields,
      startMs: task.start - base,
      endMs: task.end - base,
      durationMs: task.end - task.start,
    });
  }
  tasks.sort((a, b) => a.startMs - b.startMs);
  const activeTasks: Array<{ output: string; sourceFiles?: string[]; sourceFilesTruncated?: boolean }> = [];
  let activeBytes = 0;
  if (!completed) {
    for (const task of active.values()) {
      const sources = taskSources(task);
      const entry = {
        output: task.label,
        ...(sources.sourceFiles ? { sourceFiles: sources.sourceFiles } : {}),
        ...(sources.sourceFilesTruncated ? { sourceFilesTruncated: true } : {}),
      };
      const size = Buffer.byteLength(JSON.stringify(entry), 'utf8');
      if (activeTasks.length >= MAX_ACTIVE_TASKS || activeBytes + size > 8 * 1024) {
        truncated = true;
        break;
      }
      activeBytes += size;
      activeTasks.push(entry);
    }
  }
  // A failed/cancelled build may have incomplete tasks; never fabricate an end time.
  return {
    source: 'msbuild-file-log',
    clock: 'msbuild-cl-events',
    tasks,
    ...(!completed && activeTasks.length ? { activeTasks } : {}),
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
          return parseMsbuildCompileLog(
            buffer.subarray(0, end).toString('utf8'),
            stat.size > MAX_LOG_BYTES,
            completed,
            project.projectRoot,
            build,
          );
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
