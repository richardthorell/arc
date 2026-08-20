import fs from 'node:fs';
import path from 'node:path';

import type { ProjectTextFile } from '../common/editorWorkflowTypes';

export type BuiltinTextReaderEnvironment = {
  environmentRoot?: string;
  resourcesPath?: string;
  cwd?: string;
};

const firstExistingDirectory = (candidates: Array<string | undefined>): string | null =>
  candidates.find((candidate): candidate is string =>
    Boolean(candidate && fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()),
  ) ?? null;

export const resolveBuiltinAssetsRoot = ({
  environmentRoot,
  resourcesPath,
  cwd = process.cwd(),
}: BuiltinTextReaderEnvironment = {}): string | null =>
  firstExistingDirectory([
    environmentRoot,
    resourcesPath ? path.join(resourcesPath, 'assets') : undefined,
    resourcesPath ? path.join(resourcesPath, 'share', 'arc', 'assets') : undefined,
    path.resolve(cwd, '..', 'assets'),
    path.resolve(cwd, 'assets'),
  ]);

const relativeBuiltinPath = (assetPath: string): string => {
  const normalized = assetPath.trim().replaceAll('\\', '/').replace(/^\/+/, '');
  const relative = normalized.startsWith('builtin/') ? normalized.slice('builtin/'.length) : normalized;
  if (
    !relative ||
    relative === '..' ||
    relative.startsWith('../') ||
    path.posix.isAbsolute(relative) ||
    /^[a-z]:/i.test(relative)
  )
    throw new Error('Built-in asset path must be relative to the engine asset root');
  return path.posix.normalize(relative);
};

const pathEscapesRoot = (root: string, candidate: string): boolean => {
  const relative = path.relative(root, candidate);
  return relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative);
};

export const readBuiltinTextFile = (
  assetPath: string,
  environment: BuiltinTextReaderEnvironment = {},
): ProjectTextFile => {
  const configuredRoot = resolveBuiltinAssetsRoot(environment);
  if (!configuredRoot) throw new Error('ARC built-in assets are unavailable');

  const root = fs.realpathSync(configuredRoot);
  const relative = relativeBuiltinPath(assetPath);
  const candidate = path.resolve(root, relative);
  if (pathEscapesRoot(root, candidate)) throw new Error('Built-in asset path escapes the engine asset root');
  if (!fs.existsSync(candidate)) throw new Error(`Built-in asset does not exist: builtin/${relative}`);

  const target = fs.realpathSync(candidate);
  if (pathEscapesRoot(root, target)) throw new Error('Built-in asset resolves outside the engine asset root');
  const stats = fs.statSync(target);
  if (!stats.isFile() || stats.size > 8 * 1024 * 1024)
    throw new Error('Built-in text asset is unavailable or too large');

  return {
    path: `builtin/${relative.replaceAll('\\', '/')}`,
    text: fs.readFileSync(target, 'utf8'),
    modifiedAt: stats.mtime.toISOString(),
  };
};
