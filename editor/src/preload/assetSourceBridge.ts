import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { copyFile, mkdir, rename, rm, stat, writeFile } from 'node:fs/promises';
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
  ArcRemoteAssetKind,
} from '../common/assetSourceTypes';
import type { ArcProjectBrowserSnapshot } from '../common/projectTypes';
import { AssetSourceRegistry } from '../main/assetSources/assetSourceRegistry';
import { PolyHavenAssetSource } from '../main/assetSources/polyHavenAssetSource';
import { isSupportedModelPath } from './externalModelImport';
import { isSupportedTexturePath } from './externalTextureImport';

type Invoke = <T>(channel: string, ...args: unknown[]) => Promise<T>;
type ProgressCallback = (progress: ArcAssetImportProgress) => void;

type ImportedHostAsset = {
  guid: string;
  path: string;
  typeId: string;
  state: 'unknown' | 'queued' | 'importing' | 'ready' | 'stale' | 'failed';
  diagnostic?: string;
};

type ProjectAssetsResponse = {
  succeeded: boolean;
  error?: string;
  payload?: { assets?: ImportedHostAsset[] };
};

const imageExtensions = new Set(['exr', 'hdr', 'jpeg', 'jpg', 'png', 'tga', 'tif', 'tiff', 'webp']);
const maxMetadataResponseBytes = 64 * 1024 * 1024;
const importWaitTimeoutMs = 15_000;
const importPollIntervalMs = 75;

const normalizeSegment = (value: string): string => {
  const normalized = value
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/^\.+/, '')
    .replace(/^\.+$/, '_');
  return normalized || '_';
};

const normalizedAssetPath = (value: string): string => value.replaceAll('\\', '/').toLocaleLowerCase();

const remoteImportCanceledError = (): Error => {
  const error = new Error('Remote asset import canceled');
  error.name = 'AbortError';
  return error;
};

const throwIfAborted = (signal: AbortSignal): void => {
  if (signal.aborted) throw remoteImportCanceledError();
};

const sleep = async (milliseconds: number, signal: AbortSignal): Promise<void> => {
  throwIfAborted(signal);
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, milliseconds);
    const onAbort = () => {
      clearTimeout(timer);
      reject(remoteImportCanceledError());
    };
    signal.addEventListener('abort', onAbort, { once: true });
  });
};

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

const fileExtension = (file: ArcAssetDownloadFile): string =>
  remoteFileName(file).split('.').at(-1)?.toLocaleLowerCase() ?? '';

export const remoteDestinationPath = (
  assetId: string,
  file: ArcAssetDownloadFile,
  kind: ArcRemoteAssetKind = 'other',
): string => {
  const assetRoot = normalizeSegment(assetId);
  const filename = remoteFileName(file);
  if (kind === 'model' && imageExtensions.has(fileExtension(file))) return path.join(assetRoot, 'textures', filename);
  return path.join(assetRoot, filename);
};

const ensureContained = (root: string, candidate: string): string => {
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(candidate);
  const relative = path.relative(resolvedRoot, resolved);
  if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error('Remote asset path escapes its ARC storage root');
  }
  return resolved;
};

const httpClient = (url: URL): typeof http | typeof https => {
  if (url.protocol === 'https:') return https;
  if (url.protocol === 'http:') return http;
  throw new Error('Remote asset URL must use HTTP(S)');
};

const requestJson = async (url: string, headers: Record<string, string>, redirects = 0): Promise<unknown> => {
  if (redirects > 5) throw new Error('Asset source metadata request exceeded redirect limit');
  const parsed = new URL(url);
  const client = httpClient(parsed);
  return new Promise<unknown>((resolve, reject) => {
    const request = client.get(parsed, { headers: { Accept: 'application/json', ...headers } }, (response) => {
      const statusCode = response.statusCode ?? 0;
      if (statusCode >= 300 && statusCode < 400 && response.headers.location) {
        response.resume();
        const redirect = new URL(response.headers.location, parsed).toString();
        void requestJson(redirect, headers, redirects + 1).then(resolve, reject);
        return;
      }
      if (statusCode < 200 || statusCode >= 300) {
        response.resume();
        reject(new Error(`Asset source metadata request failed (${statusCode})`));
        return;
      }
      const chunks: Buffer[] = [];
      let bytes = 0;
      response.on('data', (chunk: Buffer) => {
        bytes += chunk.byteLength;
        if (bytes > maxMetadataResponseBytes) {
          response.destroy(new Error('Asset source metadata response is too large'));
          return;
        }
        chunks.push(chunk);
      });
      response.on('error', reject);
      response.on('end', () => {
        try {
          resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown);
        } catch (error) {
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      });
    });
    request.on('error', reject);
  });
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
  signal: AbortSignal,
  redirects = 0,
): Promise<void> => {
  if (redirects > 5) throw new Error('Remote asset download exceeded redirect limit');
  throwIfAborted(signal);
  const parsed = new URL(url);
  const client = httpClient(parsed);

  await new Promise<void>((resolve, reject) => {
    const request = client.get(parsed, { headers: { 'User-Agent': userAgent } }, (response) => {
      const statusCode = response.statusCode ?? 0;
      if (statusCode >= 300 && statusCode < 400 && response.headers.location) {
        response.resume();
        const redirect = new URL(response.headers.location, parsed).toString();
        void requestDownload(redirect, target, userAgent, onBytes, signal, redirects + 1).then(resolve, reject);
        return;
      }
      if (statusCode < 200 || statusCode >= 300) {
        response.resume();
        reject(new Error(`Remote asset download failed (${statusCode})`));
        return;
      }
      response.on('data', (chunk: Buffer) => onBytes(chunk.byteLength));
      void pipeline(response, createWriteStream(target), { signal }).then(() => resolve(), reject);
    });

    const onAbort = () => request.destroy(remoteImportCanceledError());
    signal.addEventListener('abort', onAbort, { once: true });
    request.on('close', () => signal.removeEventListener('abort', onAbort));
    request.on('error', reject);
  });
};

const downloadToCache = async (
  file: ArcAssetDownloadFile,
  cachePath: string,
  userAgent: string,
  onBytes: (bytes: number) => void,
  signal: AbortSignal,
): Promise<'cached' | 'downloaded'> => {
  throwIfAborted(signal);
  if (await cachedFileIsValid(cachePath, file)) return 'cached';
  await mkdir(path.dirname(cachePath), { recursive: true });
  const temporary = `${cachePath}.part-${process.pid}-${Date.now()}`;
  try {
    await requestDownload(file.url, temporary, userAgent, onBytes, signal);
    throwIfAborted(signal);
    if (!(await cachedFileIsValid(temporary, file))) {
      throw new Error(`Checksum verification failed for ${file.logicalPath}`);
    }
    await rm(cachePath, { force: true });
    await rename(temporary, cachePath);
    return 'downloaded';
  } finally {
    await rm(temporary, { force: true }).catch(() => undefined);
  }
};

const cacheKey = (file: ArcAssetDownloadFile): string =>
  file.checksum?.value ?? createHash('sha256').update(file.url).digest('hex');

const isImportableAssetPath = (value: string): boolean => isSupportedModelPath(value) || isSupportedTexturePath(value);

const waitForImportedAssets = async (
  invoke: Invoke,
  importedFiles: readonly string[],
  signal: AbortSignal,
): Promise<string[]> => {
  const expected = importedFiles.filter(isImportableAssetPath);
  if (expected.length === 0) {
    throw new Error("The selected remote variant does not contain a format supported by ARC's importer/cooker");
  }

  const deadline = Date.now() + importWaitTimeoutMs;
  while (Date.now() < deadline) {
    throwIfAborted(signal);
    const response = await invoke<ProjectAssetsResponse | undefined>('host:query', 'project.assets', {});
    if (response?.succeeded && response.payload?.assets) {
      const assetsByPath = new Map(response.payload.assets.map((asset) => [normalizedAssetPath(asset.path), asset]));
      const resolved = expected.map((relativePath) => assetsByPath.get(normalizedAssetPath(relativePath)));
      const failed = resolved.find((asset) => asset?.state === 'failed');
      if (failed) throw new Error(failed.diagnostic || `ARC importer failed for ${failed.path}`);
      if (resolved.every((asset) => asset?.state === 'ready')) {
        return resolved.flatMap((asset) => (asset ? [asset.guid] : []));
      }
    }
    await sleep(importPollIntervalMs, signal);
  }

  throw new Error(`Timed out waiting for ARC importer/cooker: ${expected.join(', ')}`);
};

const pathExists = async (value: string): Promise<boolean> => {
  try {
    await stat(value);
    return true;
  } catch {
    return false;
  }
};

export const createAssetSourceBridge = (invoke: Invoke) => {
  let registryPromise: Promise<{ registry: AssetSourceRegistry; userAgent: string }> | null = null;
  let nextImportOperationId = 1;
  const activeImports = new Map<number, AbortController>();

  const registry = (): Promise<{ registry: AssetSourceRegistry; userAgent: string }> => {
    if (!registryPromise) {
      registryPromise = invoke<string>('app:getVersion').then((version) => {
        const userAgent = `ARC-Editor/${version || 'dev'}`;
        return {
          registry: new AssetSourceRegistry([new PolyHavenAssetSource({ userAgent, fetchJson: requestJson })]),
          userAgent,
        };
      });
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
    return { projectRoot, contentRoot, savedRoot };
  };

  return {
    list: async (): Promise<ArcAssetSourceDescriptor[]> => (await registry()).registry.list(),
    search: async (sourceId: string, query?: ArcAssetSourceQuery): Promise<ArcAssetSearchResult> =>
      (await registry()).registry.search(sourceId, query),
    manifest: async (sourceId: string, assetId: string): Promise<ArcAssetDownloadManifest> =>
      (await registry()).registry.getDownloadManifest(sourceId, assetId),
    createImportOperation: (): number => nextImportOperationId++,
    cancelImport: (operationId: number): boolean => {
      const controller = activeImports.get(operationId);
      if (!controller) return false;
      controller.abort();
      return true;
    },
    importToProject: async (
      request: ArcAssetImportRequest,
      onProgress?: ProgressCallback,
    ): Promise<ArcAssetImportResult> => {
      if (request.destinationScope !== 'project') throw new Error('Only project-scope online imports are implemented');
      const operationId = request.operationId ?? nextImportOperationId++;
      if (!Number.isSafeInteger(operationId) || operationId <= 0)
        throw new Error('Remote import operation ID is invalid');
      if (activeImports.has(operationId)) throw new Error(`Remote import operation ${operationId} is already running`);

      const controller = new AbortController();
      const { signal } = controller;
      activeImports.set(operationId, controller);

      let stagingRoot = '';
      let destinationRoot = '';
      let provenancePath = '';
      let published = false;

      try {
        const source = await registry();
        throwIfAborted(signal);
        const asset = await source.registry.getAsset(request.sourceId, request.assetId);
        if (!asset) throw new Error(`Remote asset '${request.assetId}' no longer exists`);
        onProgress?.({ phase: 'resolving', completedFiles: 0, totalFiles: 0, completedBytes: 0 });
        const manifest = await source.registry.getDownloadManifest(request.sourceId, request.assetId);
        throwIfAborted(signal);
        const requestedPaths = new Set(request.logicalPaths);
        const selected = manifest.files.filter(
          (file) => requestedPaths.size === 0 || requestedPaths.has(file.logicalPath),
        );
        if (selected.length === 0) throw new Error('The selected remote asset variant has no files');
        if (requestedPaths.size > 0 && selected.length !== requestedPaths.size) {
          throw new Error(
            'The selected remote asset variant changed before import. Refresh the variants and try again.',
          );
        }

        const roots = await projectRoots();
        const totalBytes = selected.every((file) => file.sizeBytes !== undefined)
          ? selected.reduce((sum, file) => sum + (file.sizeBytes ?? 0), 0)
          : undefined;
        let completedBytes = 0;
        let completedFiles = 0;
        let cacheHits = 0;
        let downloadedFiles = 0;
        const importedFiles: string[] = [];
        const sourceRoot = ensureContained(
          roots.contentRoot,
          path.join(roots.contentRoot, 'External', normalizeSegment(request.sourceId)),
        );
        destinationRoot = ensureContained(sourceRoot, path.join(sourceRoot, normalizeSegment(request.assetId)));
        if (await pathExists(destinationRoot)) {
          throw new Error('This remote asset already exists in the project. Remove it before importing it again.');
        }
        await mkdir(sourceRoot, { recursive: true });
        const stagingParent = ensureContained(
          roots.savedRoot,
          path.join(roots.savedRoot, 'AssetImports', 'staging', normalizeSegment(request.sourceId)),
        );
        await mkdir(stagingParent, { recursive: true });
        stagingRoot = ensureContained(
          stagingParent,
          path.join(stagingParent, `.${normalizeSegment(request.assetId)}.import-${operationId}-${Date.now()}`),
        );
        await rm(stagingRoot, { recursive: true, force: true });
        await mkdir(stagingRoot, { recursive: true });

        for (const file of selected) {
          throwIfAborted(signal);
          const relativePath = remoteDestinationPath(request.assetId, file, asset.kind);
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
          const cacheState = await downloadToCache(
            file,
            cachePath,
            source.userAgent,
            (bytes) => {
              completedBytes += bytes;
              onProgress?.({
                phase: 'downloading',
                completedFiles,
                totalFiles: selected.length,
                completedBytes,
                totalBytes,
                currentFile: file.logicalPath,
              });
            },
            signal,
          );
          if (cacheState === 'cached') {
            cacheHits += 1;
            completedBytes += file.sizeBytes ?? 0;
          } else {
            downloadedFiles += 1;
          }
          throwIfAborted(signal);
          onProgress?.({
            phase: 'verifying',
            completedFiles,
            totalFiles: selected.length,
            completedBytes,
            totalBytes,
            currentFile: file.logicalPath,
          });
          if (!(await cachedFileIsValid(cachePath, file))) {
            throw new Error(`Cached download failed verification: ${file.logicalPath}`);
          }

          const finalPath = ensureContained(
            roots.contentRoot,
            path.join(roots.contentRoot, 'External', normalizeSegment(request.sourceId), relativePath),
          );
          const relativeToAssetRoot = path.relative(destinationRoot, finalPath);
          const stagedPath = ensureContained(stagingRoot, path.join(stagingRoot, relativeToAssetRoot));
          await mkdir(path.dirname(stagedPath), { recursive: true });
          onProgress?.({
            phase: 'staging',
            completedFiles,
            totalFiles: selected.length,
            completedBytes,
            totalBytes,
            currentFile: file.logicalPath,
          });
          await copyFile(cachePath, stagedPath);
          importedFiles.push(path.relative(roots.projectRoot, finalPath).replaceAll('\\', '/'));
          completedFiles += 1;
        }

        throwIfAborted(signal);
        onProgress?.({
          phase: 'publishing',
          completedFiles,
          totalFiles: selected.length,
          completedBytes,
          totalBytes,
        });
        await rename(stagingRoot, destinationRoot);
        stagingRoot = '';
        published = true;

        throwIfAborted(signal);
        onProgress?.({
          phase: 'importing',
          completedFiles,
          totalFiles: selected.length,
          completedBytes,
          totalBytes,
        });
        const importedAssetIds = await waitForImportedAssets(invoke, importedFiles, signal);

        const provenance = {
          sourceId: request.sourceId,
          sourceAssetId: request.assetId,
          importedAt: new Date().toISOString(),
          license: asset.license,
          sourceUrl: `${source.registry.list().find((entry) => entry.id === request.sourceId)?.homepage ?? ''}/a/${request.assetId}`,
          sourceRevision: typeof asset.metadata.filesHash === 'string' ? asset.metadata.filesHash : undefined,
        };
        provenancePath = ensureContained(
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
        throwIfAborted(signal);
        await writeFile(
          provenancePath,
          JSON.stringify(
            { provenance, importedFiles, importedAssetIds, logicalPaths: selected.map((file) => file.logicalPath) },
            null,
            2,
          ),
          'utf8',
        );

        onProgress?.({
          phase: 'complete',
          completedFiles,
          totalFiles: selected.length,
          completedBytes,
          totalBytes,
        });
        return {
          succeeded: true,
          operationId,
          destinationRoot,
          importedFiles,
          importedAssetIds,
          cacheHits,
          downloadedFiles,
          provenance,
        };
      } catch (error) {
        if (stagingRoot) await rm(stagingRoot, { recursive: true, force: true }).catch(() => undefined);
        if (published && destinationRoot) {
          await rm(destinationRoot, { recursive: true, force: true }).catch(() => undefined);
        }
        if (provenancePath) await rm(provenancePath, { force: true }).catch(() => undefined);
        if (signal.aborted) throw remoteImportCanceledError();
        throw error;
      } finally {
        activeImports.delete(operationId);
      }
    },
  };
};

export type ArcAssetSourceBridge = ReturnType<typeof createAssetSourceBridge>;
