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

type ProjectAssetsSnapshot = {
  assets?: Array<{
    guid?: string;
    path: string;
    kind?: string;
    generation?: number;
  }>;
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

const thumbnailCache = new Map<string, Promise<string | null>>();
let queue: Promise<void> = Promise.resolve();

const persistentThumbnailDatabase = 'arc-material-thumbnails-v1';
const persistentThumbnailStore = 'thumbnails';
let persistentDatabasePromise: Promise<IDBDatabase | null> | null = null;

const openPersistentThumbnailDatabase = (): Promise<IDBDatabase | null> => {
  if (typeof indexedDB === 'undefined') return Promise.resolve(null);
  if (persistentDatabasePromise) return persistentDatabasePromise;

  persistentDatabasePromise = new Promise((resolve) => {
    try {
      const request = indexedDB.open(persistentThumbnailDatabase, 1);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains(persistentThumbnailStore)) {
          database.createObjectStore(persistentThumbnailStore);
        }
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

const normalizedGuid = (guid: string) => guid.trim().replace(/[^a-zA-Z0-9-]/g, '');

// Kept stable for callers/tests that still use the old preview identity when
// diagnosing material viewport rendering. Material thumbnails no longer create
// a live viewport; they use the same native asset thumbnail renderer as pickers.
export const materialThumbnailViewportId = (guid: string, instance: number) =>
  `asset-preview-material-${normalizedGuid(guid)}~thumbnail-${instance}`;

export const materialThumbnailCacheKey = (guid: string, generation = 0, maxSize = 128) =>
  `${normalizedGuid(guid)}:${generation}:${Math.max(32, Math.min(256, Math.round(maxSize)))}`;

const colorDistance = (pixels: Uint8ClampedArray, offset: number, background: readonly number[]) =>
  Math.max(
    Math.abs(pixels[offset] - background[0]),
    Math.abs(pixels[offset + 1] - background[1]),
    Math.abs(pixels[offset + 2] - background[2]),
  );

/**
 * Removes a uniform preview clear color while preserving an isolated object.
 * The flood fill only enters pixels connected to an image edge, so matching
 * colors enclosed by the object remain opaque.
 */
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

/**
 * The native material preview uses a centered sphere with radius 0.82 in NDC.
 * Masking from that known geometry gives us a clean transparent background even
 * though the native studio background is intentionally graded instead of flat.
 */
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

const materialPathForGuid = async (guid: string): Promise<{ path: string; generation?: number } | null> => {
  const response = (await window.arc.host.query('project.assets')) as HostResponse<ProjectAssetsSnapshot>;
  if (!response.succeeded) return null;
  const normalized = normalizedGuid(guid).toLocaleLowerCase();
  const asset = response.payload?.assets?.find(
    (candidate) => normalizedGuid(candidate.guid ?? '').toLocaleLowerCase() === normalized,
  );
  return asset ? { path: asset.path, generation: asset.generation } : null;
};

const renderMaterialThumbnail = async ({ guid, maxSize = 128 }: MaterialThumbnailRequest): Promise<string | null> => {
  if (!guid || !window.arc?.host) return null;

  const outputSize = Math.max(32, Math.min(256, Math.round(maxSize)));
  const renderSize = Math.min(256, outputSize * 2);
  const asset = await materialPathForGuid(guid);
  if (!asset?.path) return null;

  const response = (await window.arc.host.query('asset.thumbnail', {
    path: asset.path,
    maxSize: renderSize,
  })) as HostResponse<AssetThumbnailSnapshot>;
  const dataUrl = response.succeeded ? response.payload?.dataUrl : null;
  if (!dataUrl) return null;

  const image = await loadImage(dataUrl);
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
  return output.toDataURL('image/png');
};

export function loadMaterialSphereThumbnail(request: MaterialThumbnailRequest): Promise<string | null> {
  const key = materialThumbnailCacheKey(request.guid, request.generation, request.maxSize);
  for (const cachedKey of thumbnailCache.keys()) {
    if (cachedKey.startsWith(`${normalizedGuid(request.guid)}:`) && cachedKey !== key) thumbnailCache.delete(cachedKey);
  }

  const cached = thumbnailCache.get(key);
  if (cached) return cached;

  let resolveTask!: (value: string | null) => void;
  const task = new Promise<string | null>((resolve) => {
    resolveTask = resolve;
  });
  thumbnailCache.set(key, task);

  queue = queue
    .catch(() => undefined)
    .then(async () => {
      try {
        const persisted = await readPersistentThumbnail(key);
        if (persisted) {
          resolveTask(persisted);
          return;
        }

        const result = await renderMaterialThumbnail(request);
        if (!result) {
          thumbnailCache.delete(key);
        } else {
          await writePersistentThumbnail(key, result);
        }
        resolveTask(result);
      } catch {
        thumbnailCache.delete(key);
        resolveTask(null);
      }
    });
  return task;
}

export function invalidateMaterialSphereThumbnail(guid: string) {
  const prefix = `${normalizedGuid(guid)}:`;
  for (const key of thumbnailCache.keys()) if (key.startsWith(prefix)) thumbnailCache.delete(key);
  void deletePersistentThumbnailPrefix(prefix);
}
