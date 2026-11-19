import fs from 'node:fs';
import path from 'node:path';

import type { WorkflowArtifact } from '../common/projectWorkflowTypes';

const within = (root: string, target: string): boolean => {
  const relative = path.relative(root, target);
  return Boolean(relative) && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
};

/**
 * Resolve an artifact selected from a persisted, project-scoped build run.
 * The renderer cannot ask Electron to reveal arbitrary local files.
 */
export function resolveBuildArtifactPath(
  projectRoot: string,
  artifacts: readonly WorkflowArtifact[] | undefined,
  requestedPath: string,
): string {
  if (!requestedPath || typeof requestedPath !== 'string') throw new Error('Invalid build artifact path');
  if (!artifacts?.some((entry) => entry.path === requestedPath || entry.manifestPath === requestedPath))
    throw new Error('Artifact does not belong to the selected build');

  const root = fs.realpathSync(projectRoot);
  const candidate = path.isAbsolute(requestedPath) ? path.normalize(requestedPath) : path.resolve(root, requestedPath);
  if (!within(root, candidate)) throw new Error('Artifact path escapes the active project');

  const real = fs.realpathSync(candidate);
  if (!within(root, real)) throw new Error('Artifact path resolves outside the active project');
  return real;
}
