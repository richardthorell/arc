import fs from 'node:fs';
import path from 'node:path';

import type { BuildTargetGraph } from '../common/buildTargetTypes';
import type { ArcProjectCandidate } from '../common/projectTypes';
import { CMakeTargetOutputObserver, readCMakeTargetGraph, requestCMakeCodemodel } from './cmakeTargetGraph';

/**
 * The normal arc-project CLI uses <project paths.build>/default; it does not
 * accept a generator build directory from the editor Build button.
 * Never walk outside the active project or follow a symlink at a build-root
 * component when writing a File API query.
 */
export function editorCommandBuildDirectory(project: ArcProjectCandidate): string | null {
  const buildRoot = project.descriptor.paths.build;
  if (typeof buildRoot !== 'string' || !buildRoot) return null;
  const root = path.resolve(project.projectRoot);
  const directory = path.resolve(root, buildRoot, 'default');
  const relative = path.relative(root, directory);
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) return null;
  let current = root;
  for (const part of relative.split(path.sep)) {
    current = path.join(current, part);
    try {
      const info = fs.lstatSync(current);
      if (info.isSymbolicLink() || !info.isDirectory()) return null;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') return null;
    }
  }
  return directory;
}

/**
 * A default CLI build requests CMake's "all" target, which the codemodel does
 * not represent as a normal named target. Only present the explicitly declared
 * editor module subtree, not a fabricated project-wide dependency graph.
 */
export class EditorCommandTargetGraph {
  private graph: BuildTargetGraph | null = null;
  private observer: CMakeTargetOutputObserver | null = null;

  private constructor(
    private readonly directory: string,
    private readonly configuration: string,
    private readonly rootTarget: string,
    private readonly notify: (graph: BuildTargetGraph) => void,
  ) {}

  static forProject(
    project: ArcProjectCandidate,
    configuration: string,
    notify: (graph: BuildTargetGraph) => void,
  ): EditorCommandTargetGraph | null {
    const directory = editorCommandBuildDirectory(project);
    const module = project.descriptor.modules?.find((item) => item.enabled && item.kind === 'editor');
    if (!directory || !module?.target) return null;
    return new EditorCommandTargetGraph(directory, configuration, module.target, notify);
  }

  prepare(): void {
    try {
      requestCMakeCodemodel(this.directory);
      // Existing configured builds have a valid reply before the next command.
      this.refresh();
    } catch {
      // The CLI's success or failure remains authoritative when telemetry fails.
    }
  }

  observe(text: string, stream: 'stdout' | 'stderr'): void {
    if (!this.graph) this.refresh();
    this.observer?.accept(text, stream);
  }

  finish(): void {
    // A configure command, or a first build that configured implicitly, may
    // have generated its first codemodel reply without producing artifact logs.
    this.refresh(true);
  }

  private refresh(force = false): void {
    try {
      const next = readCMakeTargetGraph(this.directory, this.configuration, this.rootTarget);
      if (!next) return;
      if (this.graph) {
        if (!force) return;
        const previous = this.graph;
        const signature = (graph: BuildTargetGraph) =>
          JSON.stringify(graph.nodes.map((node) => [node.id, node.name, node.type, node.dependencies, node.artifacts]));
        if (signature(next) === signature(previous)) return;
        next.revision = previous.revision + 1;
        // A reconfigure may have changed target outputs; never transfer stale
        // observations to a changed artifact or dependency definition.
        for (const node of next.nodes) {
          const old = previous.nodes.find((item) => item.id === node.id);
          if (
            old &&
            JSON.stringify([old.dependencies, old.artifacts]) === JSON.stringify([node.dependencies, node.artifacts])
          )
            node.artifactObservedAt = old.artifactObservedAt;
        }
      }
      this.graph = next;
      this.observer = new CMakeTargetOutputObserver(this.directory, next, (id) => {
        const node = next.nodes.find((item) => item.id === id);
        if (!node || node.artifactObservedAt) return;
        node.artifactObservedAt = new Date().toISOString();
        next.revision++;
        this.notify(structuredClone(next));
      });
      this.notify(structuredClone(next));
    } catch {
      // Corrupt/unsupported File API replies must never fail compilation.
    }
  }
}
