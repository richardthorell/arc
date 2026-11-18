import fs from 'node:fs';
import path from 'node:path';

import type { BuildCompileTrace } from '../common/buildCompileTypes';
import type { BuildTargetGraph } from '../common/buildTargetTypes';
import type { ArcProjectCandidate } from '../common/projectTypes';
import { editorCommandBuildDirectory } from './editorCommandTargetGraph';

const MAX_DATABASE_BYTES = 8 * 1024 * 1024;
const MAX_ENTRIES = 25_000;
const MAX_TRACE_BYTES = 32 * 1024;

type CommandEntry = {
  directory?: unknown;
  file?: unknown;
  output?: unknown;
  arguments?: unknown;
  command?: unknown;
};

type SourceAssociation = { sourceFile: string; targetName?: string };

/** Resolve only source/output paths explicitly recorded by CMake's compile database. */
function within(root: string, candidate: string): string | null {
  const relative = path.relative(root, candidate);
  if (!relative || relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative)) return null;
  return relative.replace(/\\/g, '/');
}

function objectOutput(entry: CommandEntry): string | null {
  if (typeof entry.output === 'string' && entry.output.length < 2048) return entry.output;
  if (Array.isArray(entry.arguments)) {
    const args = entry.arguments;
    for (let i = 0; i < args.length; i++) {
      if (args[i] === '-o' && typeof args[i + 1] === 'string') return args[i + 1];
      if (typeof args[i] === 'string' && args[i].startsWith('-o') && args[i].length > 2) return args[i].slice(2);
    }
  }
  if (typeof entry.command !== 'string' || entry.command.length > 16_384) return null;
  // CMake's Ninja compile_commands.json contains a shell-escaped -o argument.
  // Do not infer outputs from an unrecognized compiler command.
  const match = /(?:^|\s)-o\s+(?:"([^"]+)"|'([^']+)'|([^\s]+))/u.exec(entry.command);
  return match?.[1] ?? match?.[2] ?? match?.[3] ?? null;
}

function association(entry: CommandEntry, buildRoot: string, projectRoot: string): {
  outputKey: string;
  value: SourceAssociation;
} | null {
  if (typeof entry.directory !== 'string' || entry.directory.length > 2048) return null;
  if (typeof entry.file !== 'string' || entry.file.length > 2048) return null;
  const output = objectOutput(entry);
  if (!output || output.length > 2048) return null;
  const directory = path.resolve(entry.directory);
  const file = path.resolve(directory, entry.file);
  const object = path.resolve(directory, output);
  const sourceFile = within(projectRoot, file);
  const relativeOutput = within(buildRoot, object);
  if (!sourceFile || !relativeOutput) return null;
  const target = /(?:^|\/)CMakeFiles\/([^/]+)\.dir\//u.exec(relativeOutput)?.[1];
  return {
    outputKey: object.replace(/\\/g, '/').toLowerCase(),
    value: {
      sourceFile,
      ...(target && target.length <= 128 ? { targetName: target } : {}),
    },
  };
}

/**
 * BM7.3: join Ninja's measured object outputs with CMake's compilation database.
 *
 * No names are derived from filename suffixes, guessed based on CMake graph
 * dependencies, or attributed to an unrelated native MSBuild CL task.
 * Ambiguous compilation-database entries are intentionally left unattributed.
 */
export function enrichCompileTrace(
  project: ArcProjectCandidate,
  trace: BuildCompileTrace,
  graph?: BuildTargetGraph,
): BuildCompileTrace {
  if (trace.source !== 'ninja-log' || !trace.tasks.length) return trace;
  const buildRoot = editorCommandBuildDirectory(project);
  if (!buildRoot) return trace;
  try {
    const filename = path.join(buildRoot, 'compile_commands.json');
    const info = fs.lstatSync(filename);
    if (!info.isFile() || info.isSymbolicLink() || info.size > MAX_DATABASE_BYTES) return trace;
    const parsed: unknown = JSON.parse(fs.readFileSync(filename, 'utf8'));
    if (!Array.isArray(parsed) || parsed.length > MAX_ENTRIES) return trace;
    const sources = new Map<string, SourceAssociation | null>();
    const projectRoot = path.resolve(project.projectRoot);
    for (const entry of parsed) {
      if (!entry || typeof entry !== 'object' || Array.isArray(entry)) continue;
      const record = association(entry as CommandEntry, buildRoot, projectRoot);
      if (!record) continue;
      if (sources.has(record.outputKey)) {
        // Multiple source files can emit an identical object name in malformed
        // or custom build databases. Do not attach an arbitrary source file.
        const previous = sources.get(record.outputKey);
        if (previous && JSON.stringify(previous) !== JSON.stringify(record.value)) sources.set(record.outputKey, null);
      } else {
        sources.set(record.outputKey, record.value);
      }
    }
    const verifiedNames = new Set(graph?.nodes.map((node) => node.name) ?? []);
    let bytes = Buffer.byteLength(JSON.stringify(trace), 'utf8');
    const tasks = trace.tasks.map((task) => {
      const outputKey = path.resolve(buildRoot, task.output).replace(/\\/g, '/').toLowerCase();
      const source = sources.get(outputKey);
      if (!source) return task;
      const fields = {
        sourceFile: source.sourceFile,
        ...(source.targetName && verifiedNames.has(source.targetName) ? { targetName: source.targetName } : {}),
      };
      const extra = Buffer.byteLength(JSON.stringify(fields), 'utf8') + 2;
      if (bytes + extra > MAX_TRACE_BYTES) return task;
      bytes += extra;
      return { ...task, ...fields };
    });
    return { ...trace, tasks };
  } catch {
    // Missing, invalid, or unsafe metadata must never affect native build results.
    return trace;
  }
}
