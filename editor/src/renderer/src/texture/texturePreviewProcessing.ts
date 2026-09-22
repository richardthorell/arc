import type { TextureSettingsSnapshot } from './textureSettings';
export { evaluateTextureCurve, processTexturePixel } from './texturePixelProcessing';
import TexturePreviewWorker from './texturePreviewWorker?worker';

export type TexturePreviewMode = 'source' | 'processed' | 'difference';
export type TextureHistogram = { r: number[]; g: number[]; b: number[]; a: number[] };
export type TexturePreviewAnalysis = {
  width: number;
  height: number;
  sourcePixels: Uint8ClampedArray;
  histogram: TextureHistogram;
};

let cachedSource: Promise<Omit<TexturePreviewAnalysis, 'histogram'>> | null = null;
let cachedUrl: string | null = null;

async function decodeSource(dataUrl: string): Promise<Omit<TexturePreviewAnalysis, 'histogram'>> {
  const image = new Image();
  image.src = dataUrl;
  await image.decode();
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas is unavailable');
  context.drawImage(image, 0, 0);
  return {
    width: canvas.width,
    height: canvas.height,
    sourcePixels: context.getImageData(0, 0, canvas.width, canvas.height).data,
  };
}

export async function analyzeTexturePreview(
  dataUrl: string,
  settings: TextureSettingsSnapshot,
  signal?: AbortSignal,
): Promise<TexturePreviewAnalysis> {
  if (dataUrl !== cachedUrl || !cachedSource) {
    cachedUrl = dataUrl;
    cachedSource = decodeSource(dataUrl);
  }
  const source = await cachedSource;
  if (signal?.aborted) throw new DOMException('Analysis cancelled', 'AbortError');
  // A bounded sample is sufficient for the histogram; full-resolution per-pixel work belongs on the GPU.
  const stride = Math.max(1, Math.ceil(Math.max(source.width, source.height) / 512));
  const samples = new Uint8ClampedArray(Math.ceil(source.width / stride) * Math.ceil(source.height / stride) * 4);
  let destination = 0;
  for (let y = 0; y < source.height; y += stride) {
    for (let x = 0; x < source.width; x += stride) {
      const offset = (y * source.width + x) * 4;
      samples.set(source.sourcePixels.subarray(offset, offset + 4), destination);
      destination += 4;
    }
  }
  const histogram = await new Promise<TextureHistogram>((resolve, reject) => {
    const worker = new TexturePreviewWorker();
    const abort = () => {
      worker.terminate();
      reject(new DOMException('Analysis cancelled', 'AbortError'));
    };
    signal?.addEventListener('abort', abort, { once: true });
    worker.onmessage = (event: MessageEvent<TextureHistogram>) => {
      signal?.removeEventListener('abort', abort);
      worker.terminate();
      resolve(event.data);
    };
    worker.onerror = (event) => {
      signal?.removeEventListener('abort', abort);
      worker.terminate();
      reject(new Error(event.message));
    };
    worker.postMessage({ samples, settings }, [samples.buffer]);
  });
  return {
    ...source,
    histogram,
  };
}
