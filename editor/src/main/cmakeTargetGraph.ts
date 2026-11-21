import fs from 'node:fs';
import path from 'node:path';

import type { BuildTargetGraph, BuildTargetNode } from '../common/buildTargetTypes';

const MAX_TARGETS = 128;
const MAX_CONFIG_TARGETS = 8_000;
const MAX_REPLY_SIZE = 12 * 1024 * 1024;
const MAX_FILE_SIZE = 4 * 1024 * 1024;
const HIDDEN_GENERATOR_TARGETS = new Set(['zero_check']);

type FileApiTargetEntry = { id: string; name: string; jsonFile: string };
type FileApiConfiguration = { name: string; targets: FileApiTargetEntry[] };
type FileApiTarget = {
  id: string;
  name: string;
  type: string;
  dependencies?: Array<{ id: string }>;
  artifacts?: Array<{ path: string }>;
  paths?: { source?: string; build?: string };
};

/** Generator bookkeeping is not a user/module build task. */
function visibleTarget(name: string): boolean {
  return !HIDDEN_GENERATOR_TARGETS.has(name.toLowerCase());
}

function replyDirectory(buildDir: string): string {
  return path.join(buildDir, '.cmake', 'api', 'v1', 'reply');
}

/** Request native target/dependency metadata before configure; no console-output heuristics. */
export function requestCMakeCodemodel(buildDir: string): void {
  const queryDir = path.join(buildDir, '.cmake', 'api', 'v1', 'query');
  fs.mkdirSync(queryDir, { recursive: true });
  fs.writeFileSync(path.join(queryDir, 'codemodel-v2'), '');
}

function replyJson(replyDir: string, name: string, maxBytes = MAX_FILE_SIZE): unknown {
  if (!/^[A-Za-z0-9_.-]+\.json$/u.test(name)) throw new Error('Invalid CMake File API filename');
  const filename = path.join(replyDir, name);
  const info = fs.lstatSync(filename);
  if (!info.isFile() || info.isSymbolicLink() || info.size > maxBytes) throw new Error('Invalid CMake File API reply');
  return JSON.parse(fs.readFileSync(filename, 'utf8')) as unknown;
}

/**
 * Load the requested root target's reachable dependency subgraph from CMake's
 * codemodel-v2 reply. Missing/stale replies return null rather than inventing
 * target relationships. No file path is accepted from a renderer.
 */
export function readCMakeTargetGraph(
  buildDir: string,
  configuration: string,
  rootTargetName: string,
): BuildTargetGraph | null {
  const replyDir = replyDirectory(buildDir);
  if (!fs.existsSync(replyDir)) return null;
  const indices = fs
    .readdirSync(replyDir)
    .filter((name) => /^index-[\w.-]+\.json$/u.test(name))
    .sort()
    .reverse();
  if (!indices.length) return null;
  const index = replyJson(replyDir, indices[0], MAX_REPLY_SIZE) as {
    cmake?: { generator?: { name?: string } };
    reply?: Record<string, { jsonFile?: string }>;
  };
  const modelFile = index.reply?.['codemodel-v2']?.jsonFile;
  if (!modelFile) return null;
  const model = replyJson(replyDir, modelFile, MAX_REPLY_SIZE) as {
    configurations?: FileApiConfiguration[];
  };
  const config = model.configurations?.find(
    (entry) =>
      entry.name.toLowerCase() === configuration.toLowerCase() ||
      // Single-configuration CMake generators (Ninja/Unix Makefiles) use "".
      (entry.name === '' && model.configurations?.length === 1),
  );
  if (!config || !Array.isArray(config.targets) || config.targets.length > MAX_CONFIG_TARGETS) return null;

  const byId = new Map<string, FileApiTargetEntry>();
  for (const target of config.targets) {
    if (typeof target.id === 'string' && typeof target.jsonFile === 'string' && visibleTarget(target.name))
      byId.set(target.id, target);
  }
  const root = config.targets.find((target) => target.name === rootTargetName);
  if (!root || !byId.has(root.id)) return null;

  const nodes: BuildTargetNode[] = [];
  const visited = new Set<string>();
  const pending = [root.id];
  while (pending.length && nodes.length < MAX_TARGETS) {
    const id = pending.shift()!;
    if (visited.has(id)) continue;
    visited.add(id);
    const entry = byId.get(id);
    if (!entry) continue;
    const detail = replyJson(replyDir, entry.jsonFile) as FileApiTarget;
    if (detail.id !== id || detail.name !== entry.name) throw new Error('CMake File API target identity mismatch');
    const dependencies = (detail.dependencies ?? [])
      .map((dependency) => dependency.id)
      .filter((dependency) => byId.has(dependency));
    const artifacts = (detail.artifacts ?? [])
      .map((artifact) => artifact.path)
      .filter((artifact) => typeof artifact === 'string' && artifact.length <= 1024);
    const sourceDirectory =
      typeof detail.paths?.source === 'string' && detail.paths.source.length <= 1024 ? detail.paths.source : undefined;
    const buildDirectory =
      typeof detail.paths?.build === 'string' && detail.paths.build.length <= 1024 ? detail.paths.build : undefined;
    nodes.push({
      id,
      name: entry.name,
      type: detail.type,
      dependencies,
      artifacts,
      ...(sourceDirectory ? { sourceDirectory } : {}),
      ...(buildDirectory ? { buildDirectory } : {}),
    });
    for (const dependency of dependencies) if (!visited.has(dependency)) pending.push(dependency);
  }

  const included = new Set(nodes.map((node) => node.id));
  for (const node of nodes) node.dependencies = node.dependencies.filter((id) => included.has(id));
  return {
    revision: 1,
    generator: index.cmake?.generator?.name ?? 'CMake',
    configuration: config.name || configuration,
    rootTargetId: root.id,
    discoveredAt: new Date().toISOString(),
    nodes,
    truncated: pending.length > 0,
  };
}

function normalizedArtifact(buildDir: string, filename: string): string {
  const result = path.isAbsolute(filename) ? filename : path.resolve(buildDir, filename);
  return result.replace(/\\/g, '/').replace(/\/+$/u, '').toLowerCase();
}

/**
 * Observe only native build output lines that identify a *specific* artifact
 * already declared in the File API graph (e.g. "X.vcxproj -> C:\\...\\X.dll").
 * An incremental build can skip targets: missing output does not mean failure.
 */
export class CMakeTargetOutputObserver {
  private readonly artifacts = new Map<string, string>();
  private readonly observed = new Set<string>();
  private pendingStdout = '';
  private pendingStderr = '';

  constructor(
    buildDir: string,
    graph: BuildTargetGraph,
    private readonly onArtifact: (targetId: string) => void,
  ) {
    for (const node of graph.nodes) {
      for (const artifact of node.artifacts) this.artifacts.set(normalizedArtifact(buildDir, artifact), node.id);
    }
    this.buildDir = buildDir;
  }

  private readonly buildDir: string;

  accept(text: string, stream: 'stdout' | 'stderr'): void {
    const combined = (stream === 'stdout' ? this.pendingStdout : this.pendingStderr) + text;
    const lines = combined.split(/\r?\n/u);
    // Do not retain unbounded partial lines from a tool that prints without newlines.
    const rest = (lines.pop() ?? '').slice(-8192);
    if (stream === 'stdout') this.pendingStdout = rest;
    else this.pendingStderr = rest;
    for (const line of lines) this.observeLine(line);
  }

  private observeLine(line: string): void {
    const arrow = line.lastIndexOf(' -> ');
    if (arrow < 0) return;
    const candidate = line
      .slice(arrow + 4)
      .trim()
      .replace(/^["']|["']$/gu, '');
    if (!candidate) return;
    const id = this.artifacts.get(normalizedArtifact(this.buildDir, candidate));
    if (!id || this.observed.has(id)) return;
    this.observed.add(id);
    this.onArtifact(id);
  }
}
