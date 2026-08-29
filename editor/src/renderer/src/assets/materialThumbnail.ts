type HostResponse<T = unknown> = {
  succeeded: boolean;
  error?: string;
  payload?: T;
};

type ViewportState = {
  submitted?: boolean;
  frameIndex?: number;
  assetPreviewError?: string;
};

type MaterialThumbnailRequest = {
  guid: string;
  generation?: number;
  maxSize?: number;
};

type PixelBounds = {
  x: number;
  y: number;
  width: number;
  height: number;
};

const thumbnailCache = new Map<string, Promise<string | null>>();
let queue: Promise<void> = Promise.resolve();
let thumbnailInstance = 0;

const sleep = (milliseconds: number) => new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));

const normalizedGuid = (guid: string) => guid.trim().replace(/[^a-zA-Z0-9-]/g, '');

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
 * Removes the uniform preview clear color while preserving the isolated sphere.
 * The flood fill only enters pixels connected to an image edge, so a material
 * that happens to contain the clear color inside the sphere remains opaque.
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

const queryViewportState = async (viewportId: string) =>
  (await window.arc.host.query('viewport.state', { viewportId })) as HostResponse<ViewportState>;

const waitForPreviewFrame = async (viewportId: string, minimumFrameIndex: number) => {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    const response = await queryViewportState(viewportId);
    if (response.succeeded && response.payload?.assetPreviewError) throw new Error(response.payload.assetPreviewError);
    if (
      response.succeeded &&
      response.payload?.submitted &&
      (response.payload.frameIndex ?? 0) >= minimumFrameIndex
    ) {
      // Let the shared-texture receiver copy the completed native frame into the hidden canvas.
      await sleep(75);
      return;
    }
    await sleep(25);
  }
  throw new Error('Timed out while rendering the material thumbnail');
};

const renderMaterialThumbnail = async ({ guid, maxSize = 128 }: MaterialThumbnailRequest): Promise<string | null> => {
  if (!guid || !window.arc?.host || !window.arc?.viewport) return null;

  const outputSize = Math.max(32, Math.min(256, Math.round(maxSize)));
  const renderSize = Math.min(512, outputSize * 2);
  const instance = ++thumbnailInstance;
  const viewportId = materialThumbnailViewportId(guid, instance);
  const canvas = document.createElement('canvas');
  canvas.id = `arc-material-thumbnail-${instance}`;
  canvas.width = renderSize;
  canvas.height = renderSize;
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, {
    position: 'absolute',
    left: '-10000px',
    top: '0',
    width: `${renderSize}px`,
    height: `${renderSize}px`,
    pointerEvents: 'none',
  });
  document.body.appendChild(canvas);
  window.arc.viewport.registerSurface(viewportId, canvas.id);

  try {
    await window.arc.viewport.create({ viewportId, x: 0, y: 0, width: renderSize, height: renderSize });
    const stateBeforeOptions = await queryViewportState(viewportId);
    const baselineFrameIndex = stateBeforeOptions.payload?.frameIndex ?? 0;
    await window.arc.host.command('viewport.setRenderOptions', {
      viewportId,
      renderMode: 'shaded',
      visualization: 'standard',
      overlay: 'none',
      shadows: true,
      grid: false,
      realtime: true,
      cameraSpeed: 1,
      antiAliasing: 'disabled',
      environment: {
        sky: false,
        fog: false,
        terrain: false,
        water: false,
        vegetation: false,
        decals: false,
      },
    });
    // Surface creation can publish before the material-preview scene has propagated
    // to the shared texture. Capture a few realtime frames later so the thumbnail
    // reflects the realized material instead of the primitive fallback.
    await waitForPreviewFrame(viewportId, baselineFrameIndex + 3);

    const context = canvas.getContext('2d');
    if (!context) return null;
    const frame = context.getImageData(0, 0, renderSize, renderSize);
    const transparent = transparentPreviewPixels(frame.data, renderSize, renderSize);

    const source = document.createElement('canvas');
    source.width = renderSize;
    source.height = renderSize;
    const sourceContext = source.getContext('2d');
    if (!sourceContext) return null;
    const transparentImage = sourceContext.createImageData(renderSize, renderSize);
    transparentImage.data.set(transparent);
    sourceContext.putImageData(transparentImage, 0, 0);

    const output = document.createElement('canvas');
    output.width = outputSize;
    output.height = outputSize;
    const outputContext = output.getContext('2d');
    if (!outputContext) return null;
    outputContext.clearRect(0, 0, outputSize, outputSize);
    outputContext.imageSmoothingEnabled = true;
    outputContext.imageSmoothingQuality = 'high';

    const bounds = opaquePixelBounds(transparent, renderSize, renderSize);
    if (bounds) {
      const availableSize = outputSize * 0.84;
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
  } catch {
    return null;
  } finally {
    window.arc.viewport.unregisterSurface(viewportId);
    await window.arc.viewport.detach(viewportId).catch(() => undefined);
    canvas.remove();
  }
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
      const result = await renderMaterialThumbnail(request);
      if (!result) thumbnailCache.delete(key);
      resolveTask(result);
    });
  return task;
}

export function invalidateMaterialSphereThumbnail(guid: string) {
  const prefix = `${normalizedGuid(guid)}:`;
  for (const key of thumbnailCache.keys()) if (key.startsWith(prefix)) thumbnailCache.delete(key);
}
