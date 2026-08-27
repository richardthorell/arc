import fs from 'node:fs';
import path from 'node:path';

import type { ArcProjectCandidate } from '../common/projectTypes';

export const supportedTextureExtensions = new Set([
  '.bmp',
  '.exr',
  '.hdr',
  '.jpeg',
  '.jpg',
  '.png',
  '.tga',
  '.webp',
]);

export type ExternalTextureImportResult = {
  path: string;
  sourcePath: string;
};

const normalizeRelativePath = (value: string) => value.replaceAll('\\', '/').replace(/^\/+/, '');

export const isSupportedTexturePath = (value: string): boolean =>
  supportedTextureExtensions.has(path.extname(value).toLocaleLowerCase());

const projectContainedPath = (projectRoot: string, relativePath: string): string => {
  const root = fs.realpathSync(projectRoot);
  const resolved = path.resolve(root, normalizeRelativePath(relativePath));
  const relative = path.relative(root, resolved);
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative))
    throw new Error('Texture destination escapes the active project');
  return resolved;
};

const uniqueTexturePath = (directory: string, filename: string): string => {
  const extension = path.extname(filename);
  const stem = path.basename(filename, extension);
  let candidate = path.join(directory, filename);
  for (let suffix = 1; fs.existsSync(candidate); ++suffix) {
    candidate = path.join(directory, `${stem}_${suffix}${extension}`);
  }
  return candidate;
};

export const importExternalTexture = (
  sourcePath: string,
  project: ArcProjectCandidate,
): ExternalTextureImportResult => {
  if (!project.writable) throw new Error('The active project is read-only');
  if (!sourcePath || !path.isAbsolute(sourcePath)) throw new Error('Dropped texture path is invalid');
  if (!isSupportedTexturePath(sourcePath)) throw new Error(`Unsupported texture format: ${path.extname(sourcePath)}`);

  const sourceStats = fs.statSync(sourcePath);
  if (!sourceStats.isFile()) throw new Error('Dropped texture is not a file');

  const contentRoot = normalizeRelativePath(project.descriptor.paths.content || 'Content');
  const textureDirectoryRelative = normalizeRelativePath(path.posix.join(contentRoot, 'Textures'));
  const textureDirectory = projectContainedPath(project.projectRoot, textureDirectoryRelative);
  fs.mkdirSync(textureDirectory, { recursive: true });

  const destination = uniqueTexturePath(textureDirectory, path.basename(sourcePath));
  fs.copyFileSync(sourcePath, destination, fs.constants.COPYFILE_EXCL);

  return {
    sourcePath,
    path: normalizeRelativePath(path.relative(project.projectRoot, destination)),
  };
};
