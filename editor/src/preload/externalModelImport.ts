import fs from 'node:fs';
import path from 'node:path';

import type { ArcProjectCandidate } from '../common/projectTypes';

export const supportedModelExtensions = new Set(['.fbx', '.glb', '.gltf', '.obj']);

export type ExternalModelDependency = {
  path: string;
  sourcePath: string;
  kind: 'buffer' | 'material' | 'texture' | 'other';
  exists: boolean;
};

export type ExternalModelImportPlan = {
  sourcePath: string;
  fileName: string;
  dependencies: ExternalModelDependency[];
};

export type ExternalModelImportResult = {
  path: string;
  sourcePath: string;
  importedDependencies: string[];
};

export const isSupportedModelPath = (value: string): boolean =>
  supportedModelExtensions.has(path.extname(value).toLocaleLowerCase());

const isExternalReference = (value: string): boolean =>
  Boolean(value) && !/^(?:data:|https?:|file:)/iu.test(value) && !path.isAbsolute(value);

const normalizedDependency = (
  sourceDirectory: string,
  reference: string,
  kind: ExternalModelDependency['kind'],
): ExternalModelDependency | null => {
  if (!isExternalReference(reference)) return null;
  const clean = decodeURIComponent(reference.split(/[?#]/u, 1)[0]).replaceAll('\\', '/');
  if (!clean || clean === '.' || clean === '..' || clean.startsWith('../')) return null;
  const sourcePath = path.resolve(sourceDirectory, clean);
  const relative = path.relative(sourceDirectory, sourcePath).replaceAll('\\', '/');
  if (!relative || relative === '..' || relative.startsWith('../') || path.isAbsolute(relative)) return null;
  return { path: relative, sourcePath, kind, exists: fs.existsSync(sourcePath) && fs.statSync(sourcePath).isFile() };
};

const dedupeDependencies = (items: Array<ExternalModelDependency | null>): ExternalModelDependency[] => {
  const deduped = new Map<string, ExternalModelDependency>();
  for (const item of items) {
    if (!item) continue;
    const key = item.sourcePath.toLocaleLowerCase();
    if (!deduped.has(key)) deduped.set(key, item);
  }
  return [...deduped.values()].sort((left, right) => left.path.localeCompare(right.path));
};

const scanGltfDependencies = (sourcePath: string): ExternalModelDependency[] => {
  const sourceDirectory = path.dirname(sourcePath);
  const document = JSON.parse(fs.readFileSync(sourcePath, 'utf8')) as {
    buffers?: Array<{ uri?: string }>;
    images?: Array<{ uri?: string }>;
  };
  return dedupeDependencies([
    ...(document.buffers ?? []).map((buffer) =>
      buffer.uri ? normalizedDependency(sourceDirectory, buffer.uri, 'buffer') : null,
    ),
    ...(document.images ?? []).map((image) =>
      image.uri ? normalizedDependency(sourceDirectory, image.uri, 'texture') : null,
    ),
  ]);
};

const mtlTextureReference = /^(?:map_[A-Za-z0-9_]+|bump|disp|decal|refl)\s+(.+)$/iu;

const scanMtlDependencies = (sourceDirectory: string, mtlPath: string): ExternalModelDependency[] => {
  if (!fs.existsSync(mtlPath) || !fs.statSync(mtlPath).isFile()) return [];
  const mtlDirectory = path.dirname(mtlPath);
  const dependencies: Array<ExternalModelDependency | null> = [];
  for (const rawLine of fs.readFileSync(mtlPath, 'utf8').split(/\r?\n/u)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const match = mtlTextureReference.exec(line);
    if (!match) continue;
    const tokens = match[1].trim().split(/\s+/u);
    const reference = tokens.at(-1) ?? '';
    const dependency = normalizedDependency(mtlDirectory, reference, 'texture');
    if (!dependency) continue;
    const relativeToModel = path.relative(sourceDirectory, dependency.sourcePath).replaceAll('\\', '/');
    if (relativeToModel === '..' || relativeToModel.startsWith('../') || path.isAbsolute(relativeToModel)) continue;
    dependencies.push({ ...dependency, path: relativeToModel });
  }
  return dedupeDependencies(dependencies);
};

const scanObjDependencies = (sourcePath: string): ExternalModelDependency[] => {
  const sourceDirectory = path.dirname(sourcePath);
  const dependencies: Array<ExternalModelDependency | null> = [];
  for (const rawLine of fs.readFileSync(sourcePath, 'utf8').split(/\r?\n/u)) {
    const line = rawLine.trim();
    if (!line.toLocaleLowerCase().startsWith('mtllib ')) continue;
    for (const reference of line.slice(7).trim().split(/\s+/u)) {
      const material = normalizedDependency(sourceDirectory, reference, 'material');
      if (!material) continue;
      dependencies.push(material);
      dependencies.push(...scanMtlDependencies(sourceDirectory, material.sourcePath));
    }
  }
  return dedupeDependencies(dependencies);
};

export const analyzeExternalModel = (sourcePath: string): ExternalModelImportPlan => {
  if (!path.isAbsolute(sourcePath)) throw new Error('External model source path must be absolute');
  if (!fs.existsSync(sourcePath) || !fs.statSync(sourcePath).isFile())
    throw new Error('External model source file does not exist');
  if (!isSupportedModelPath(sourcePath))
    throw new Error(`Unsupported model format: ${path.extname(sourcePath) || 'unknown'}`);

  const extension = path.extname(sourcePath).toLocaleLowerCase();
  const dependencies =
    extension === '.gltf'
      ? scanGltfDependencies(sourcePath)
      : extension === '.obj'
        ? scanObjDependencies(sourcePath)
        : [];
  return { sourcePath, fileName: path.basename(sourcePath), dependencies };
};

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

const normalizedProjectFolder = (projectRoot: string, contentRoot: string, requestedFolder?: string): string => {
  const relative = (requestedFolder?.trim() || `${contentRoot}/Models`).replaceAll('\\', '/').replace(/^\/+/, '');
  if (!relative || relative === '..' || relative.startsWith('../') || path.isAbsolute(relative))
    throw new Error('Model import destination must be project-relative');
  const normalized = path.normalize(relative);
  if (normalized === '..' || normalized.startsWith(`..${path.sep}`))
    throw new Error('Model import destination escapes the project');
  const contentRelative = normalized.replaceAll('\\', '/');
  if (contentRelative !== contentRoot && !contentRelative.startsWith(`${contentRoot}/`))
    throw new Error('Models can only be imported into the project content folder');
  return path.join(projectRoot, normalized);
};

export const importExternalModel = (
  sourcePath: string,
  project: ArcProjectCandidate,
  requestedFolder?: string,
  selectedDependencies?: readonly string[],
): ExternalModelImportResult => {
  if (!project.writable) throw new Error('The active project is read-only');
  if (!path.isAbsolute(sourcePath)) throw new Error('External model source path must be absolute');
  if (!fs.existsSync(sourcePath) || !fs.statSync(sourcePath).isFile())
    throw new Error('External model source file does not exist');
  if (!isSupportedModelPath(sourcePath))
    throw new Error(`Unsupported model format: ${path.extname(sourcePath) || 'unknown'}`);

  const projectRoot = fs.realpathSync(project.projectRoot);
  const contentRoot = (project.descriptor.paths.content || 'Content').replaceAll('\\', '/').replace(/^\/+|\/+$/g, '');
  const destinationDirectory = normalizedProjectFolder(projectRoot, contentRoot, requestedFolder);
  fs.mkdirSync(destinationDirectory, { recursive: true });
  const destination = ensureUniqueDestination(destinationDirectory, path.basename(sourcePath));
  fs.copyFileSync(sourcePath, destination, fs.constants.COPYFILE_EXCL);

  const plan = analyzeExternalModel(sourcePath);
  const selected = new Set(
    selectedDependencies ?? plan.dependencies.filter((item) => item.exists).map((item) => item.path),
  );
  const importedDependencies: string[] = [];
  for (const dependency of plan.dependencies) {
    if (!dependency.exists || !selected.has(dependency.path)) continue;
    const dependencyDestination = path.resolve(destinationDirectory, dependency.path);
    const relativeDestination = path.relative(destinationDirectory, dependencyDestination);
    if (
      !relativeDestination ||
      relativeDestination === '..' ||
      relativeDestination.startsWith(`..${path.sep}`) ||
      path.isAbsolute(relativeDestination)
    )
      continue;
    fs.mkdirSync(path.dirname(dependencyDestination), { recursive: true });
    fs.copyFileSync(dependency.sourcePath, dependencyDestination, fs.constants.COPYFILE_EXCL);
    importedDependencies.push(path.relative(projectRoot, dependencyDestination).replaceAll('\\', '/'));
  }

  return {
    path: path.relative(projectRoot, destination).replaceAll('\\', '/'),
    sourcePath: destination,
    importedDependencies,
  };
};
