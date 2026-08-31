import fs from 'node:fs';
import path from 'node:path';

import type { ProjectSnapshot } from './assetSourceBridge';

export const supportedModelExtensions = new Set(['.fbx', '.glb', '.gltf', '.obj']);

export type ExternalModelImportResult = {
  path: string;
  sourcePath: string;
};

export const isSupportedModelPath = (value: string): boolean =>
  supportedModelExtensions.has(path.extname(value).toLocaleLowerCase());

const ensureUniqueDestination = (directory: string, fileName: string): string => {
  const parsed = path.parse(fileName);
  let candidate = path.join(directory, fileName);
  let suffix = 1;
  while (fs.existsSync(candidate)) {
    candidate = path.join(directory, `${parsed.name}_${suffix}${parsed.ext}`);
    suffix += 1;
  }
  return candidate;
};

const normalizedProjectFolder = (projectRoot: string, requestedFolder?: string): string => {
  const relative = (requestedFolder?.trim() || 'Content/Models').replaceAll('\\', '/').replace(/^\/+/, '');
  if (!relative || relative === '..' || relative.startsWith('../') || path.isAbsolute(relative))
    throw new Error('Model import destination must be project-relative');
  const normalized = path.normalize(relative);
  if (normalized === '..' || normalized.startsWith(`..${path.sep}`))
    throw new Error('Model import destination escapes the project');
  const contentRelative = normalized.replaceAll('\\', '/');
  if (contentRelative !== 'Content' && !contentRelative.startsWith('Content/'))
    throw new Error('Models can only be imported into the project Content folder');
  return path.join(projectRoot, normalized);
};

export const importExternalModel = (
  sourcePath: string,
  project: ProjectSnapshot,
  requestedFolder?: string,
): ExternalModelImportResult => {
  if (!project.writable) throw new Error('The active project is read-only');
  if (!path.isAbsolute(sourcePath)) throw new Error('External model source path must be absolute');
  if (!fs.existsSync(sourcePath) || !fs.statSync(sourcePath).isFile()) throw new Error('External model source file does not exist');
  if (!isSupportedModelPath(sourcePath)) throw new Error(`Unsupported model format: ${path.extname(sourcePath) || 'unknown'}`);

  const projectRoot = fs.realpathSync(project.projectRoot);
  const destinationDirectory = normalizedProjectFolder(projectRoot, requestedFolder);
  fs.mkdirSync(destinationDirectory, { recursive: true });
  const destination = ensureUniqueDestination(destinationDirectory, path.basename(sourcePath));
  fs.copyFileSync(sourcePath, destination, fs.constants.COPYFILE_EXCL);

  return {
    path: path.relative(projectRoot, destination).replaceAll('\\', '/'),
    sourcePath: destination,
  };
};
