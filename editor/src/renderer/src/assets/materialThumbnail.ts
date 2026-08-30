type HostResponse<T = unknown> = {
  succeeded: boolean;
  error?: string;
  payload?: T;
};

type MaterialThumbnailRequest = {
  guid: string;
  generation?: number;
  maxSize?: number;
};

type ProjectAsset = {
  guid?: string;
  path: string;
  kind?: string;
  generation?: number;
};

type ProjectAssetsSnapshot = {
  assets?: ProjectAsset[];
};

type AssetThumbnailSnapshot = {
  path: string;
  width: number;
  height: number;
  dataUrl: string;
};

type PixelBounds = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type ThumbnailTrace = {
  id: string;
  startedAt: number;
  mark: (event: string, details?: Record<string, unknown>) => void;
};

const thumbnailCache = new Map<string, Promise<string | null>>();
const observedGenerations = new Map<string, number>();
const maxConcurrentThumbnailRenders = 3;
let activeThumbnailRenders = 0;
const pendingThumbnailRenders: Array<() => void> = [];
let thumbnailTraceSequence = 0;
let projectAssetsPromise: Promise<Map<string, ProjectAsset>> | null = null;

const persistentThumbnailDatabase = 'arc-material-thumbnails-v1';
const persistentThumbnailStore = 'thumbnails';
const persistentThumbnailVersion = 2;
let persistentDatabasePromise: Promise<IDBDatabase | null> | null = null;

const normalizedGuid = (guid: string) => guid.trim().replace(/[^a-zA-Z0-9-]/g, '');
const normalizedGuidKey = (guid: string) => normalizedGuid(guid).toLocaleLowerCase();
const thumbnailSize = (maxSize = 128) => Math.max(32, Math.min(256, Math.round(maxSize)));
const persistentThumbnailPrefix = (guid: string) => `v${persistentThumbnailVersion}:${normalizedGuidKey(guid)}:`;
const persistentThumbnailKey = (guid: string, maxSize = 128) =>
  `${persistentThumbnailPrefix(guid)}${thumbnailSize(maxSize)}`;

const traceNow = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const createThumbnailTrace = (request: MaterialThumbnailRequest, key: string): ThumbnailTrace => {
  const startedAt = traceNow();
  const id = `${++thumbnailTraceSequence}:${normalizedGuid(request.guid).slice(0, 8) || 'unknown'}`;
  const mark = (event: string, details: Record<string, unknown> = {}) => {
    window.console.info(`[material-thumbnail ${id}] +${(traceNow() - startedAt).toFixed(1)}ms ${event}`, {
      key,
      ...details,
    });
  };
  mark('request', {
    guid: request.guid,
    generation: request.generation ?? 0,
    maxSize: request.maxSize ?? 128,
    persistentKey: persistentThumbnailKey(request.guid, request.maxSize),
  });
  return { id, startedAt, mark };
};

const withThumbnailRenderSlot = async <T>(operation: () => Promise<T>, trace?: ThumbnailTrace): Promise<T> => {
  if (activeThumbnailRenders >= maxConcurrentThumbnailRenders) {
    trace?.mark('queue.wait', {
      active: activeThumbnailRenders,
      pending: pendingThumbnailRenders.length + 1,
    });
    await new Promise<void>((resolve) => pendingThumbnailRenders.push(resolve));
  }
  activeThumbnailRenders += 1;
  trace?.mark('queue.acquired', {
    active: activeThumbnailRenders,
    pending: pendingThumbnailRenders.length,
  });
  try {
    return await operation();
  } finally {
    activeThumbnailRenders -= 1;
    trace?.mark('queue.released', {
      active: activeThumbnailRenders,
      pending: pendingThumbnailRenders.length,
    });
    pendingThumbnailRenders.shift()?.();
  }
};

const openPersistentThumbnailDatabase = (): Promise<IDBDatabase | null> => {
  if (typeof indexedDB === 'undefined') return Promise.resolve(null);
  if (persistentDatabasePromise) return persistentDatabasePromise;

  persistentDatabasePromise = new Promise((resolve) => {
    try {
      const request = indexedDB.open(persistentThumbnailDatabase, 1);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains(persistentThumbnailStore))
          database.createObjectStore(persistentThumbnailStore);
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
      request.onblocked = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
  return persistentDatabasePromise;
};

const readPersistentThumbnail = async (key: string): Promise<string | null> => {
  const database = await openPersistentThumbnailDatabase();
  if (!database) return null;
  return new Promise((resolve) => {
    try {
      const transaction = database.transaction(persistentThumbnailStore, 'readonly');
      const request = transaction.objectStore(persistentThumbnailStore).get(key);
      request.onsuccess = () => resolve(typeof request.result === 'string' ? request.result : null);
      request.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
};

const writePersistentThumbnail = async (key: string, value: string): Promise<void> => {
  const database = await openPersistentThumbnailDatabase();
  if (!database) return;
  await new Promise<void>((resolve) => {
    try {
      const transaction = database.transaction(persistentThumbnailStore, 'readwrite');
      transaction.objectStore(persistentThumbnailStore).put(value, key);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => resolve();
      transaction.onabort = () => resolve();
    } catch {
      resolve();
    }
  });
};

const deletePersistentThumbnailPrefix = async (prefix: string): Promise<void> => {
  const database = await openPersistentThumbnailDatabase();
  if (!database) return;
  await new Promise<void>((resolve) => {
    try {
      const transaction = database.transaction(persistentThumbnailStore, 'readwrite');
      const store = transaction.objectStore(persistentThumbnailStore);
      const request = store.openCursor();
      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor) return;
        if (typeof cursor.key === 'string' && cursor.key.startsWith(prefix)) cursor.delete();
        cursor.continue();
      };
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => resolve();
      transaction.onabort = () => resolve();
    } catch {
      resolve();
    }
  });
};

const projectAssetsByGuid = async (): Promise<Map<string, ProjectAsset>> => {
  if (!projectAssetsPromise) {
    projectAssetsPromise = (async () => {
      const response = (await window.arc.host.query('project.assets')) as HostResponse<ProjectAssetsSnapshot>;
      if (!response.succeeded) return new Map<string, ProjectAsset>();
      const assets = new Map<string, ProjectAsset>();
      for (const asset of response.payload?.assets ?? []) {
        if (!asset.guid) continue;
        assets.set(normalizedGuidKey(asset.guid), asset);
      }
      return assets;
    })().catch(() => {
      projectAssetsPromise = null;
      return new Map<string, ProjectAsset>();
    });
  }
  return projectAssetsPromise;
};

const materialPathForGuid = async (guid: string): Promise<ProjectAsset | null> => {
  const key = normalizedGuidKey(guid);
  const assets = await projectAssetsByGuid();
  const cached = assets.get(key);
  if (cached) return cached;

  // The project snapshot may predate a newly imported material. Refresh only on a miss.
  projectAssetsPromise = null;
  return (await projectAssetsByGuid()).get(key) ?? null;
};

// Kept stable for callers/tests that still use the old preview identity when
// diagnosing material viewport rendering. Material thumbnails no longer create
// a live viewport; they use the same native asset thumbnail renderer as pickers.
export const materialThumbnailViewportId = (guid: string, instance: number) =>
  `asset-preview-material-${normalizedGuid(guid)}~thumbnail-${instance}`;

// Runtime generations intentionally remain part of the in-memory key so an edit
// refreshes immediately during the current editor session. Persistent entries use
// a separate stable key because runtime generations are rebuilt across startups.
export const materialThumbnailCacheKey = (guid: string, generation = 0, maxSize = 128) =>
  `${normalizedGuid(guid)}:${generation}:${thumbnailSize(maxSize)}`;

const colorDistance = (pixels: Uint8ClampedArray, offset: number, background: readonly number[]) =>
  Math.max(
    Math.abs(pixels[offset] - background[0]),
    Math.abs(pixels[offset + 1] - background[1]),
    Math.abs(pixels[offset + 2] - background[2]),
  );

export function transparentPreviewPixels(
  source: Uint8ClampedArray,
  width: number,
  height: number,
  tolerance = 12,
): Uint8ClampedArray {
  const pixels = new Uint8ClampedArray(source);
  if (width <= 0 || height <= 0 || pixels.length < width * height * 4) return pixels;

  const corners = [0, (width - 1) * 4, (height - 1) * width * 4, (height * width - 1) * 4];
  const background = [0, 1, 2].map((channel) =>
    Math.round(corners.reduce((sum, offset) => sum + pixels[offset + channel], 0) / corners.length),
  );
  const visited = new Uint8Array(width * height);
  const queuePixels = new Int32Array(width * height);
  let read = 0;
  let write = 0;

  const enqueue = (index: number) => {
    if (index < 0 || index >= visited.length || visited[index]) return;
    const offset = index * 4;
    if (colorDistance(pixels, offset, background) > tolerance) return;
    visited[index] = 1;
    queuePixels[write++] = index;
  };

  for (let x = 0; x < width; x += 1) {
    enqueue(x);
    enqueue((height - 1) * width + x);
  }
  for (let y = 1; y + 1 < height; y += 1) {
    enqueue(y * width);
    enqueue(y * width + width - 1);
  }

  while (read < write) {
    const index = queuePixels[read++];
    const x = index % width;
    const y = Math.floor(index / width);
    pixels[index * 4 + 3] = 0;
    if (x > 0) enqueue(index - 1);
    if (x + 1 < width) enqueue(index + 1);
    if (y > 0) enqueue(index - width);
    if (y + 1 < height) enqueue(index + width);
  }

  return pixels;
}

export function maskMaterialSpherePixels(
  source: Uint8ClampedArray,
  width: number,
  height: number,
  sphereRadius = 0.82,
): Uint8ClampedArray {
  const pixels = new Uint8ClampedArray(source);
  if (width <= 0 || height <= 0 || pixels.length < width * height * 4) return pixels;

  const centerX = width * 0.5;
  const centerY = height * 0.5;
  const radius = Math.min(width, height) * 0.5 * sphereRadius;
  const feather = Math.max(1, Math.min(width, height) / 192);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const dx = x + 0.5 - centerX;
      const dy = y + 0.5 - centerY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const offset = (y * width + x) * 4 + 3;
      if (distance >= radius + feather) pixels[offset] = 0;
      else if (distance > radius - feather) {
        const coverage = Math.max(0, Math.min(1, (radius + feather - distance) / (feather * 2)));
        pixels[offset] = Math.round(pixels[offset] * coverage);
      }
    }
  }
  return pixels;
}

export function opaquePixelBounds(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  minimumAlpha = 16,
): PixelBounds | null {
  if (width <= 0 || height <= 0 || pixels.length < width * height * 4) return null;

  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (pixels[(y * width + x) * 4 + 3] < minimumAlpha) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }

  if (maxX < minX || maxY < minY) return null;
  return { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}

const loadImage = (dataUrl: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Material thumbnail image could not be decoded'));
    image.src = dataUrl;
  });

const renderMaterialThumbnail = async (
  { guid, maxSize = 128 }: MaterialThumbnailRequest,
  trace?: ThumbnailTrace,
): Promise<string | null> => {
  if (!guid || !window.arc?.host) {
    trace?.mark('render.unavailable', { hasGuid: Boolean(guid), hasHost: Boolean(window.arc?.host) });
    return null;
  }

  const outputSize = thumbnailSize(maxSize);
  const renderSize = Math.min(256, outputSize * 2);
  trace?.mark('asset.resolve.start', { sharedSnapshot: Boolean(projectAssetsPromise) });
  const resolveStartedAt = traceNow();
  const asset = await materialPathForGuid(guid);
  trace?.mark('asset.resolve.end', {
    durationMs: Number((traceNow() - resolveStartedAt).toFixed(1)),
    path: asset?.path ?? null,
  });
  if (!asset?.path) return null;

  trace?.mark('native.thumbnail.start', { path: asset.path, renderSize });
  const nativeStartedAt = traceNow();
  const response = (await window.arc.host.query('asset.thumbnail', {
    path: asset.path,
    maxSize: renderSize,
  })) as HostResponse<AssetThumbnailSnapshot>;
  trace?.mark('native.thumbnail.end', {
    durationMs: Number((traceNow() - nativeStartedAt).toFixed(1)),
    succeeded: response.succeeded,
    error: response.error ?? null,
    width: response.payload?.width ?? null,
    height: response.payload?.height ?? null,
    encodedBytes: response.payload?.dataUrl?.length ?? 0,
  });
  const dataUrl = response.succeeded ? response.payload?.dataUrl : null;
  if (!dataUrl) return null;

  trace?.mark('image.decode.start');
  const decodeStartedAt = traceNow();
  const image = await loadImage(dataUrl);
  trace?.mark('image.decode.end', {
    durationMs: Number((traceNow() - decodeStartedAt).toFixed(1)),
    width: image.naturalWidth,
    height: image.naturalHeight,
  });

  const compositeStartedAt = traceNow();
  trace?.mark('composite.start');
  const sourceWidth = image.naturalWidth || response.payload?.width || renderSize;
  const sourceHeight = image.naturalHeight || response.payload?.height || renderSize;
  const source = document.createElement('canvas');
  source.width = sourceWidth;
  source.height = sourceHeight;
  const sourceContext = source.getContext('2d');
  if (!sourceContext) return null;
  sourceContext.drawImage(image, 0, 0, sourceWidth, sourceHeight);

  const frame = sourceContext.getImageData(0, 0, sourceWidth, sourceHeight);
  const transparent = maskMaterialSpherePixels(frame.data, sourceWidth, sourceHeight);
  const transparentImage = sourceContext.createImageData(sourceWidth, sourceHeight);
  transparentImage.data.set(transparent);
  sourceContext.clearRect(0, 0, sourceWidth, sourceHeight);
  sourceContext.putImageData(transparentImage, 0, 0);

  const output = document.createElement('canvas');
  output.width = outputSize;
  output.height = outputSize;
  const outputContext = output.getContext('2d');
  if (!outputContext) return null;
  outputContext.clearRect(0, 0, outputSize, outputSize);
  outputContext.imageSmoothingEnabled = true;
  outputContext.imageSmoothingQuality = 'high';

  const bounds = opaquePixelBounds(transparent, sourceWidth, sourceHeight);
  if (bounds) {
    const availableSize = outputSize * 0.82;
    const scale = Math.min(availableSize / bounds.width, availableSize / bounds.height);
    const drawWidth = bounds.width * scale;
    const drawHeight = bounds.height * scale;
    outputContext.drawImage(
      source,
      bounds.x,
      bounds.y,
      bounds.width,
      bounds.height,
      (outputSize - drawWidth) * 0.5,
      (outputSize - drawHeight) * 0.5,
      drawWidth,
      drawHeight,
    );
  } else {
    outputContext.drawImage(source, 0, 0, outputSize, outputSize);
  }

  const result = output.toDataURL('image/png');
  trace?.mark('composite.end', {
    durationMs: Number((traceNow() - compositeStartedAt).toFixed(1)),
    outputSize,
    outputBytes: result.length,
    bounds,
  });
  return result;
};

export function loadMaterialSphereThumbnail(request: MaterialThumbnailRequest): Promise<string | null> {
  const guidKey = normalizedGuidKey(request.guid);
  const generation = request.generation ?? 0;
  const key = materialThumbnailCacheKey(request.guid, generation, request.maxSize);
  const diskKey = persistentThumbnailKey(request.guid, request.maxSize);
  const trace = createThumbnailTrace(request, key);
  const previousGeneration = observedGenerations.get(guidKey);

  if (previousGeneration !== undefined && previousGeneration !== generation) {
    trace.mark('generation.changed', { previousGeneration, generation, diskKey });
    for (const cachedKey of thumbnailCache.keys()) {
      if (cachedKey.startsWith(`${normalizedGuid(request.guid)}:`)) thumbnailCache.delete(cachedKey);
    }
    void deletePersistentThumbnailPrefix(persistentThumbnailPrefix(request.guid));
    projectAssetsPromise = null;
  }
  observedGenerations.set(guidKey, generation);

  for (const cachedKey of thumbnailCache.keys()) {
    if (cachedKey.startsWith(`${normalizedGuid(request.guid)}:`) && cachedKey !== key) {
      trace.mark('memory.invalidate-old-generation', { cachedKey });
      thumbnailCache.delete(cachedKey);
    }
  }

  const cached = thumbnailCache.get(key);
  if (cached) {
    trace.mark('memory.hit');
    void cached.then((value) => trace.mark('memory.hit.resolved', { hasThumbnail: Boolean(value) }));
    return cached;
  }
  trace.mark('memory.miss');

  const task = (async () => {
    try {
      trace.mark('disk.lookup.start', { diskKey });
      const diskStartedAt = traceNow();
      const persisted = await readPersistentThumbnail(diskKey);
      trace.mark(persisted ? 'disk.hit' : 'disk.miss', {
        diskKey,
        durationMs: Number((traceNow() - diskStartedAt).toFixed(1)),
        encodedBytes: persisted?.length ?? 0,
      });
      if (persisted) {
        trace.mark('complete', { source: 'disk', totalMs: Number((traceNow() - trace.startedAt).toFixed(1)) });
        return persisted;
      }

      const result = await withThumbnailRenderSlot(() => renderMaterialThumbnail(request, trace), trace);
      if (!result) {
        thumbnailCache.delete(key);
        trace.mark('complete.empty', { totalMs: Number((traceNow() - trace.startedAt).toFixed(1)) });
        return null;
      }

      trace.mark('disk.write.start', { diskKey, encodedBytes: result.length });
      const writeStartedAt = traceNow();
      await writePersistentThumbnail(diskKey, result);
      trace.mark('disk.write.end', { diskKey, durationMs: Number((traceNow() - writeStartedAt).toFixed(1)) });
      trace.mark('complete', { source: 'render', totalMs: Number((traceNow() - trace.startedAt).toFixed(1)) });
      return result;
    } catch (error) {
      thumbnailCache.delete(key);
      trace.mark('failed', {
        totalMs: Number((traceNow() - trace.startedAt).toFixed(1)),
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  })();

  thumbnailCache.set(key, task);
  return task;
}

export function invalidateMaterialSphereThumbnail(guid: string) {
  const memoryPrefix = `${normalizedGuid(guid)}:`;
  const diskPrefix = persistentThumbnailPrefix(guid);
  window.console.info('[material-thumbnail] invalidate', { guid, memoryPrefix, diskPrefix });
  observedGenerations.delete(normalizedGuidKey(guid));
  projectAssetsPromise = null;
  for (const key of thumbnailCache.keys()) if (key.startsWith(memoryPrefix)) thumbnailCache.delete(key);
  void deletePersistentThumbnailPrefix(diskPrefix);
}
