import fs from 'node:fs';
import path from 'node:path';

import type { AgentAssetWorkspace } from './editorAgentHarness';

export type AgentProjectContext = {
  projectRoot: string;
  assetRoots: string[];
  writable: boolean;
};

export class ProjectAssetWorkspace implements AgentAssetWorkspace {
  private readonly createdPaths = new Set<string>();

  constructor(private readonly activeProject: () => AgentProjectContext | null) {}

  async exists(relativePath: string): Promise<boolean> {
    return fs.existsSync(this.resolve(relativePath));
  }

  async create(relativePath: string, contents: string): Promise<void> {
    const project = this.requireProject();
    if (!project.writable) throw new Error('The active project is read-only');
    const target = this.resolve(relativePath);
    if (fs.existsSync(target)) throw new Error(`Asset already exists: ${relativePath}`);
    const temporary = `${target}.agent-${process.pid}-${Date.now()}.tmp`;
    try {
      fs.writeFileSync(temporary, contents, { encoding: 'utf8', flag: 'wx' });
      fs.copyFileSync(temporary, target, fs.constants.COPYFILE_EXCL);
      this.createdPaths.add(target);
    } finally {
      fs.rmSync(temporary, { force: true });
    }
  }

  async remove(relativePath: string): Promise<void> {
    const target = this.resolve(relativePath);
    if (!this.createdPaths.delete(target)) throw new Error('Refusing to remove an asset not created by this workspace');
    fs.rmSync(target, { force: true });
  }

  private requireProject(): AgentProjectContext {
    const project = this.activeProject();
    if (!project) throw new Error('No project is open');
    return project;
  }

  private resolve(relativePath: string): string {
    const project = this.requireProject();
    const projectRoot = fs.realpathSync(project.projectRoot);
    const configuredRoot = project.assetRoots[0] || 'Content';
    const assetRootCandidate = path.resolve(projectRoot, configuredRoot);
    const projectRelativeRoot = path.relative(projectRoot, assetRootCandidate);
    if (
      projectRelativeRoot === '..' ||
      projectRelativeRoot.startsWith(`..${path.sep}`) ||
      path.isAbsolute(projectRelativeRoot)
    )
      throw new Error('The primary asset root escapes the active project');
    const assetRoot = fs.realpathSync(assetRootCandidate);
    const normalized = relativePath.replaceAll('\\', '/').replace(/^\/+/, '');
    if (!normalized || normalized === '..' || normalized.startsWith('../') || path.posix.isAbsolute(normalized))
      throw new Error('Asset path must be relative to the primary content root');
    const target = path.resolve(assetRoot, normalized);
    const relative = path.relative(assetRoot, target);
    if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative))
      throw new Error('Asset path escapes the primary content root');
    const containmentTarget = fs.existsSync(target) ? target : path.dirname(target);
    const realContainmentTarget = fs.realpathSync(containmentTarget);
    const realRelative = path.relative(assetRoot, realContainmentTarget);
    if (realRelative === '..' || realRelative.startsWith(`..${path.sep}`) || path.isAbsolute(realRelative))
      throw new Error('Asset path resolves outside the primary content root');
    return target;
  }
}
