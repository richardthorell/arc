import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, readFile, stat, unlink, writeFile, copyFile } from 'node:fs/promises';
import http from 'node:http';
import https from 'node:https';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';

import type {
  ArcAssetDownloadFile,
  ArcAssetDownloadManifest,
  ArcAssetImportProgress,
  ArcAssetImportRequest,
  ArcAssetImportResult,
  ArcAssetSearchResult,
  ArcAssetSourceDescriptor,
  ArcAssetSourceQuery,
} from '../common/assetSourceTypes';
import type { ArcProjectBrowserSnapshot } from '../common/projectTypes';
import { createDefaultAssetSourceRegistry, type AssetSourceRegistry } from '../main/assetSources/assetSourceRegistry';

type Invoke = <T>(channel: string, ...args: unknown[]) => Promise<T>;
type ProgressCallback = (progress: ArcAssetImportProgress) => void;

const normalizeSegment = (value: string): string => {
  const normalized = value.trim().replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^\.+$/, '_');
  return normalized || '_';
};

const safeLogicalSegments = (logicalPath: string): string[] =>
  logicalPath
    .replaceAll('\\', '/')
    .split('/')
    .filter((segment) => segment && segment !== '.' && segment !== '..')
    .map(normalizeSegment);

export const remoteFileName = (file: ArcAssetDownloadFile): string => {
  try {
    const url = new URL(file.url);
    const candidate = decodeURIComponent(url.pathname.split('/').filter(Boolean).at(-1) ?? '');
    if (candidate) return normalizeSegment(candidate);
  } catch {
    // The manifest adapter owns URL validation; retain a deterministic fallback here.
  }
  return `${normalizeSegment(file.logicalPath.replaceAll('/', '_')) || 'asset'}.bin`;
};

export const remoteDestinationPath = (assetId: string, file: ArcAssetDownloadFile): string =>
  path.join(normalizeSegment(assetId), ...safeLogicalSegments(file.logicalPath), remoteFileName(file));

const ensureContained = (root: string, candidate: string): string => {
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(candidate);
  const relative = path.relative(resolvedRoot, resolved);
  if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error('Remote asset path escapes its ARC storage root');
  }
  return resolved;
};

const hashFile = async (filePath: string, algorithm: 'md5' | 'sha256'): Promise<string> =>
  new Promise((resolve, reject) => {
    const hash = createHash(algorithm);
    const stream = createReadStream(filePath);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });

const cachedFileIsValid = async (filePath: string, file: ArcAssetDownloadFile): Promise<boolean> => {
  try {
    const fileStats = await stat(filePath);
    if (!fileStats.isFile()) return false;
    if (file.sizeBytes !== undefined && fileStats.size !== file.sizeBytes) return false;
    if (!file.checksum) return true;
    return (await hashFile(filePath, file.checksum.algorithm)).toLowerCase() === file.checksum.value.toLowerCase();
  } catch {
    return false;
  }
};

const requestDownload = async (
  url: string,
  target: string,
  userAgent: string,
  onBytes: (bytes: number) => void,
  redirects = 0,
): Promise<void> => {
  if (redirects > 5) throw new Error('Remote asset download exceeded redirect limit');
  const parsed = new URL(url);
  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') throw new Error('Remote asset URL must use HTTP(S)');
  const client = parsed.protocol === 'https:' ? https : http;

  await new Promise<void>((resolve, reject) => {
    const request = client.get(parsed, { headers: { 'User-Agent': userAgent } }, (response) => {
      const statusCode = response.statusCode ?? 0;
      if (statusCode >= 300 && statusCode < 400 && response.headers.location) {
        response.resume();
        const redirect = new URL(response.headers.location, parsed).toString();
        void requestDownload(redirect, target, userAgent, onBytes, redirects + 1).then(resolve, reject);
        return;
      }
      if (statusCode < 200 || statusCode >= 300) {
        response.resume();
        reject(new Error(`Remote asset download failed (${statusCode})`));
        return;
      }
      response.on('data', (chunk: Buffer) => onBytes(chunk.byteLength));
      void pipeline(response, createWriteStream(target)).then(() => resolve(), reject);
    });
    request.on('error', reject);
  });
};

const downloadToCache = async (
  file: ArcAssetDownloadFile,
  cachePath: string,
  userAgent: string,
  onBytes: (bytes: number) => void,
): Promise<'cached' | 'downloaded'> => {
  if (await cachedFileIsValid(cachePath, file)) return 'cached';
  await mkdir(path.dirname(cachePath), { recursive: true });
  const temporary = `${cachePath}.part-${process.pid}-${Date.now()}`;
  try {
    await requestDownload(file.url, temporary, userAgent, onBytes);
    if (!(await cachedFileIsValid(temporary, file))) throw new Error(`Checksum verification failed for ${file.logicalPath}`);
    await copyFile(temporary, cachePath);
    return 'downloaded';
  } finally {
    await unlink(temporary).catch(() => undefined);
  }
};

const cacheKey = (file: ArcAssetDownloadFile): string =>
  file.checksum?.value ?? createHash('sha256').update(file.url).digest('hex');

export const createAssetSourceBridge = (invoke: Invoke) => {
  let registryPromise: Promise<{ registry: AssetSourceRegistry; userAgent: string }> | null = null;

  const registry = (): Promise<{ registry: AssetSourceRegistry; userAgent: string }> => {
    if (!registryPromise) {
      registryPromise = invoke<string>('app:getVersion').then((version) => ({
        registry: createDefaultAssetSourceRegistry(version),
        userAgent: `ARC-Editor/${version || 'dev'}`,
      }));
    }
    return registryPromise;
  };

  const projectRoots = async () => {
    const snapshot = await invoke<ArcProjectBrowserSnapshot | null>('project:snapshot');
    const project = snapshot?.activeProject;
    if (!project) throw new Error('Open a project before importing an online asset');
    if (!project.writable) throw new Error('The active project is read-only');
    const projectRoot = path.resolve(project.projectRoot);
    const contentRoot = ensureContained(projectRoot, path.resolve(projectRoot, project.descriptor.paths.content));
    const savedRoot = ensureContained(projectRoot, path.resolve(projectRoot, project.descriptor.paths.saved));
    return { project, projectRoot, contentRoot, savedRoot };
  };

  return {
    list: async (): Promise<ArcAssetSourceDescriptor[]> => (await registry()).registry.list(),
    search: async (sourceId: string, query?: ArcAssetSourceQuery): Promise<ArcAssetSearchResult> =>
      (await registry()).registry.search(sourceId, query),
    manifest: async (sourceId: string, assetId: string): Promise<ArcAssetDownloadManifest> =>
      (await registry()).registry.getDownloadManifest(sourceId, assetId),
    importToProject: async (
      request: ArcAssetImportRequest,
      onProgress?: ProgressCallback,
    ): Promise<ArcAssetImportResult> => {
      if (request.destinationScope !== 'project') throw new Error('Only project-scope online imports are implemented');
      const source = await registry();
      const asset = await source.registry.getAsset(request.sourceId, request.assetId);
      if (!asset) throw new Error(`Remote asset '${request.assetId}' no longer exists`);
      onProgress?.({ phase: 'resolving', completedFiles: 0, totalFiles: 0, completedBytes: 0 });
      const manifest = await source.registry.getDownloadManifest(request.sourceId, request.assetId);
      const requestedPaths = new Set(request.logicalPaths);
      const selected = manifest.files.filter((file) => requestedPaths.size === 0 || requestedPaths.has(file.logicalPath));
      if (selected.length === 0) throw new Error('The selected remote asset variant has no files');

      const roots = await projectRoots();
      const totalBytes = selected.every((file) => file.sizeBytes !== undefined)
        ? selected.reduce((sum, file) => sum + (file.sizeBytes ?? 0), 0)
        : undefined;
      let completedBytes = 0;
      let completedFiles = 0;
      let cacheHits = 0;
      let downloadedFiles = 0;
      const importedFiles: string[] = [];
      const destinationRoot = ensureContained(
        roots.contentRoot,
        path.join(roots.contentRoot, 'External', normalizeSegment(request.sourceId), normalizeSegment(request.assetId)),
      );

      for (const file of selected) {
        const relativePath = remoteDestinationPath(request.assetId, file);
        const cachePath = ensureContained(
          roots.savedRoot,
          path.join(
            roots.savedRoot,
            'AssetCache',
            'Remote',
            normalizeSegment(request.sourceId),
            normalizeSegment(request.assetId),
            cacheKey(file),
            remoteFileName(file),
          ),
        );
        onProgress?.({
          phase: 'downloading',
          completedFiles,
          totalFiles: selected.length,
          completedBytes,
          totalBytes,
          currentFile: file.logicalPath,
        });
        const cacheState = await downloadToCache(file, cachePath, source.userAgent, (bytes) => {
          completedBytes += bytes;
          onProgress?.({
            phase: 'downloading',
            completedFiles,
            totalFiles: selected.length,
            completedBytes,
            totalBytes,
            currentFile: file.logicalPath,
          });
        });
        if (cacheState === 'cached') {
          cacheHits += 1;
          completedBytes += file.sizeBytes ?? 0;
        } else {
          downloadedFiles += 1;
        }
        onProgress?.({
          phase: 'verifying',
          completedFiles,
          totalFiles: selected.length,
          completedBytes,
          totalBytes,
          currentFile: file.logicalPath,
        });
        if (!(await cachedFileIsValid(cachePath, file))) throw new Error(`Cached download failed verification: ${file.logicalPath}`);
        const destinationPath = ensureContained(
          roots.contentRoot,
          path.join(roots.contentRoot, 'External', normalizeSegment(request.sourceId), relativePath),
        );
        await mkdir(path.dirname(destinationPath), { recursive: true });
        onProgress?.({
          phase: 'copying',
          completedFiles,
          totalFiles: selected.length,
          completedBytes,
          totalBytes,
          currentFile: file.logicalPath,
        });
        await copyFile(cachePath, destinationPath);
        importedFiles.push(path.relative(roots.projectRoot, destinationPath).replaceAll('\\', '/'));
        completedFiles += 1;
      }

      const provenance = {
        sourceId: request.sourceId,
        sourceAssetId: request.assetId,
        importedAt: new Date().toISOString(),
        license: asset.license,
        sourceUrl: `${source.registry.list().find((entry) => entry.id === request.sourceId)?.homepage ?? ''}/a/${request.assetId}`,
        sourceRevision: typeof asset.metadata.filesHash === 'string' ? asset.metadata.filesHash : undefined,
      };
      const provenancePath = ensureContained(
        roots.savedRoot,
        path.join(
          roots.savedRoot,
          'AssetImports',
          'provenance',
          normalizeSegment(request.sourceId),
          `${normalizeSegment(request.assetId)}.json`,
        ),
      );
      await mkdir(path.dirname(provenancePath), { recursive: true });
      await writeFile(
        provenancePath,
        JSON.stringify({ provenance, importedFiles, logicalPaths: selected.map((file) => file.logicalPath) }, null, 2),
        'utf8',
      );

      onProgress?.({
        phase: 'complete',
        completedFiles,
        totalFiles: selected.length,
        completedBytes,
        totalBytes,
      });
      return { succeeded: true, destinationRoot, importedFiles, cacheHits, downloadedFiles, provenance };
    },
  };
};

export type ArcAssetSourceBridge = ReturnType<typeof createAssetSourceBridge>;
