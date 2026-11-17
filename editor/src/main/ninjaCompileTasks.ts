import fs from 'node:fs';
import path from 'node:path';

import type { BuildCompileTask, BuildCompileTrace } from '../common/buildCompileTypes';
import type { ArcProjectCandidate } from '../common/projectTypes';
import { editorCommandBuildDirectory } from './editorCommandTargetGraph';

const MAX_LOG_BYTES = 8 * 1024 * 1024;
const MAX_TASKS = 256;
// BM2 stores run summaries in a bounded JSON index. Keep trace payloads small
// across 100 historical builds until a dedicated task-trace store exists.
const MAX_TRACE_BYTES = 24 * 1024;
const HEADER = '# ninja log v5';

/**
 * Snapshot a Ninja log cursor before invoking arc-project, then consume ONLY
 * records appended by this build. Old incremental-build rows have unrelated
 * invocation-relative clocks and must never enter the new run's timeline.
 *
 * Ninja v5 records measured start/end milliseconds for completed build edges.
 * Object outputs are compilation candidates; these are not CMake target
 * timings and cannot be aligned to the editor command's absolute start time.
 */
export function beginNinjaCompileTrace(project: ArcProjectCandidate): (() => BuildCompileTrace | undefined) | null {
  const directory = editorCommandBuildDirectory(project);
  if (!directory) return null;
  const filename = path.join(directory, '.ninja_log');
  let offset = 0;
  let identity: { dev: number; ino: number } | null = null;
  try {
    const stat = fs.lstatSync(filename);
    if (!stat.isFile() || stat.isSymbolicLink()) return null;
    offset = stat.size;
    identity = { dev: stat.dev, ino: stat.ino };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') return null;
  }

  return () => {
    let fd: number | undefined;
    try {
      fd = fs.openSync(filename, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0));
      const stat = fs.fstatSync(fd);
      if (!stat.isFile() || stat.size < offset) return undefined;
      if (identity && (stat.dev !== identity.dev || stat.ino !== identity.ino)) return undefined;
      const header = Buffer.alloc(HEADER.length);
      const headerBytes = fs.readSync(fd, header, 0, header.length, 0);
      if (headerBytes !== HEADER.length || header.toString() !== HEADER) return undefined;
      const available = stat.size - offset;
      if (!available) return undefined;
      const size = Math.min(available, MAX_LOG_BYTES);
      const data = Buffer.alloc(size);
      const read = fs.readSync(fd, data, 0, size, offset);
      const end = data.lastIndexOf(10, read - 1);
      if (end < 0) return undefined;
      const lines = data.subarray(0, end).toString('utf8').split('\n');
      const tasks: BuildCompileTask[] = [];
      let truncated = available > MAX_LOG_BYTES;
      let payloadBytes = 0;
      for (const line of lines) {
        if (!line || line.startsWith('#')) continue;
        const fields = line.trimEnd().split('\t');
        if (fields.length < 5 || !/(?:^|[/\\])[^/\\]+\.(?:o|obj)$/i.test(fields[3])) continue;
        const startMs = Number(fields[0]);
        const endMs = Number(fields[1]);
        if (!Number.isSafeInteger(startMs) || !Number.isSafeInteger(endMs)) continue;
        if (startMs < 0 || endMs < startMs) continue;
        const bytes = Buffer.byteLength(fields[3], 'utf8') + 96;
        if (tasks.length >= MAX_TASKS || payloadBytes + bytes > MAX_TRACE_BYTES) {
          truncated = true;
          break;
        }
        payloadBytes += bytes;
        tasks.push({ output: fields[3], startMs, endMs, durationMs: endMs - startMs });
      }
      return { source: 'ninja-log', clock: 'ninja-invocation', tasks, truncated };
    } catch {
      // Timing extraction is best-effort and must never change build results.
      return undefined;
    } finally {
      if (fd !== undefined) fs.closeSync(fd);
    }
  };
}
