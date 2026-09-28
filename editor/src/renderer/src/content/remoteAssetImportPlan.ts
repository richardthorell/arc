import type { ArcAssetDownloadFile, ArcAssetDownloadManifest, ArcAssetImportRequest } from '../../../common/assetSourceTypes';

export type RemoteAssetImportPlan = {
  request: ArcAssetImportRequest;
  files: ArcAssetDownloadFile[];
  totalBytes?: number;
};

const normalizedLogicalPath = (value: string): string => value.replaceAll('\\', '/').replace(/^\.\//, '');

const isSafeLogicalPath = (value: string): boolean => {
  const normalized = normalizedLogicalPath(value);
  if (!normalized || normalized.startsWith('/') || /^[A-Za-z]:\//.test(normalized)) return false;
  return normalized.split('/').every((part) => part.length > 0 && part !== '.' && part !== '..');
};

/**
 * Builds the exact import request from a provider manifest selection.
 *
 * The plan is intentionally provider-neutral: only files present in the fetched manifest may be
 * imported, duplicate selections collapse deterministically, and unsafe logical paths are rejected
 * before the request crosses the renderer/main-process boundary.
 */
export function createRemoteAssetImportPlan(
  manifest: ArcAssetDownloadManifest,
  selectedLogicalPaths: readonly string[],
): RemoteAssetImportPlan {
  const manifestFiles = new Map<string, ArcAssetDownloadFile>();
  for (const file of manifest.files) {
    const path = normalizedLogicalPath(file.logicalPath);
    if (!isSafeLogicalPath(path)) throw new Error(`Remote asset manifest contains an unsafe path: ${file.logicalPath}`);
    if (manifestFiles.has(path)) throw new Error(`Remote asset manifest contains a duplicate path: ${path}`);
    manifestFiles.set(path, file);
  }

  const files: ArcAssetDownloadFile[] = [];
  const seen = new Set<string>();
  for (const requestedPath of selectedLogicalPaths) {
    const path = normalizedLogicalPath(requestedPath);
    if (seen.has(path)) continue;
    seen.add(path);
    const file = manifestFiles.get(path);
    if (!file) throw new Error(`Selected remote asset file is not present in the manifest: ${requestedPath}`);
    files.push(file);
  }

  if (files.length === 0) throw new Error('Select at least one remote asset file to import.');

  const sizes = files.map((file) => file.sizeBytes);
  const totalBytes = sizes.every((size): size is number => size !== undefined)
    ? sizes.reduce((total, size) => total + size, 0)
    : undefined;

  return {
    request: {
      sourceId: manifest.sourceId,
      assetId: manifest.assetId,
      logicalPaths: files.map((file) => normalizedLogicalPath(file.logicalPath)),
      destinationScope: 'project',
    },
    files,
    totalBytes,
  };
}
