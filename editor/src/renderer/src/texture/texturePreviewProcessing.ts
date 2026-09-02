import type { TextureChannelSource, TextureCurve, TextureSettingsSnapshot } from './textureSettings';

export type TexturePreviewMode = 'source' | 'processed' | 'difference';
export type TextureHistogram = { r: number[]; g: number[]; b: number[]; a: number[] };
export type TexturePreviewAnalysis = {
  width: number;
  height: number;
  sourceDataUrl: string;
  processedDataUrl: string;
  differenceDataUrl: string;
  sourcePixels: Uint8ClampedArray;
  processedPixels: Uint8ClampedArray;
  histogram: TextureHistogram;
};

const clamp = (value: number, low = 0, high = 1) => Math.min(high, Math.max(low, value));
const srgbToLinear = (value: number) => (value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4);
const linearToSrgb = (value: number) => {
  const v = clamp(value);
  return v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055;
};
const curveSlope = (a: TextureCurve[number], b: TextureCurve[number]) => (b.x > a.x ? (b.y - a.y) / (b.x - a.x) : 0);
const curveTangent = (curve: TextureCurve, i: number) => {
  if (i === 0) return curveSlope(curve[0], curve[1]);
  if (i === curve.length - 1) return curveSlope(curve[i - 1], curve[i]);
  const l = curveSlope(curve[i - 1], curve[i]),
    r = curveSlope(curve[i], curve[i + 1]);
  return l * r <= 0 ? 0 : (l + r) / 2;
};
export const evaluateTextureCurve = (curve: TextureCurve, input: number) => {
  const x = clamp(input);
  const ri = curve.findIndex((p) => p.x > x);
  if (ri < 0) return curve.at(-1)?.y ?? x;
  if (ri === 0) return curve[0].y;
  const li = ri - 1,
    a = curve[li],
    b = curve[ri],
    w = Math.max(1e-6, b.x - a.x),
    t = (x - a.x) / w;
  if (a.interpolation === 'constant') return a.y;
  if (a.interpolation === 'linear') return a.y + (b.y - a.y) * t;
  const m0 = a.interpolation === 'manual' ? a.outTangent : curveTangent(curve, li),
    m1 = b.interpolation === 'manual' ? b.inTangent : curveTangent(curve, ri),
    t2 = t * t,
    t3 = t2 * t;
  return clamp(
    (2 * t3 - 3 * t2 + 1) * a.y + (t3 - 2 * t2 + t) * w * m0 + (-2 * t3 + 3 * t2) * b.y + (t3 - t2) * w * m1,
  );
};
const mapped = (rgba: readonly number[], source: TextureChannelSource) => {
  if (source === 'red') return rgba[0];
  if (source === 'green') return rgba[1];
  if (source === 'blue') return rgba[2];
  if (source === 'alpha') return rgba[3];
  return source === 'one' ? 1 : 0;
};

export function processTexturePixel(
  rgba: readonly number[],
  settings: TextureSettingsSnapshot,
): [number, number, number, number] {
  const channels = [settings.channelR, settings.channelG, settings.channelB, settings.channelA] as const;
  const inversions = [settings.invertR, settings.invertG, settings.invertB, settings.invertA] as const;
  const value = channels.map((source, index) => {
    const mappedValue = mapped(rgba, source);
    return inversions[index] ? 1 - mappedValue : mappedValue;
  });
  if (settings.semantic !== 'normal') {
    const range = Math.max(0.0001, settings.inputWhite - settings.inputBlack);
    for (let channel = 0; channel < 3; channel += 1) {
      if (settings.colorSpace === 'srgb') value[channel] = srgbToLinear(value[channel]);
      value[channel] = clamp((value[channel] - settings.inputBlack) / range);
      if (settings.curvesEnabled) {
        const curve = channel === 0 ? settings.curveR : channel === 1 ? settings.curveG : settings.curveB;
        value[channel] = evaluateTextureCurve(settings.curveMaster, evaluateTextureCurve(curve, value[channel]));
      }
      value[channel] = clamp(value[channel]) ** (1 / settings.gamma);
      value[channel] *= 2 ** settings.brightness;
      value[channel] = (value[channel] - 0.5) * settings.contrast + 0.5;
    }
    const luminance = value[0] * 0.2126 + value[1] * 0.7152 + value[2] * 0.0722;
    for (let channel = 0; channel < 3; channel += 1)
      value[channel] = luminance + (value[channel] - luminance) * settings.saturation;
    const spread = clamp(Math.max(value[0], value[1], value[2]) - Math.min(value[0], value[1], value[2]));
    const vibrance = 1 + settings.vibrance * (1 - spread);
    for (let channel = 0; channel < 3; channel += 1)
      value[channel] = luminance + (value[channel] - luminance) * vibrance;
    value[0] *= settings.tintR;
    value[1] *= settings.tintG;
    value[2] *= settings.tintB;
    for (let channel = 0; channel < 3; channel += 1) {
      value[channel] = settings.outputBlack + clamp(value[channel]) * (settings.outputWhite - settings.outputBlack);
      value[channel] = settings.colorSpace === 'srgb' ? linearToSrgb(value[channel]) : clamp(value[channel]);
    }
  }
  if (settings.curvesEnabled) value[3] = evaluateTextureCurve(settings.curveA, value[3]);
  return [clamp(value[0]), clamp(value[1]), clamp(value[2]), clamp(value[3])];
}

const emptyHistogram = (): TextureHistogram => ({
  r: Array(256).fill(0),
  g: Array(256).fill(0),
  b: Array(256).fill(0),
  a: Array(256).fill(0),
});

export async function analyzeTexturePreview(
  dataUrl: string,
  settings: TextureSettingsSnapshot,
): Promise<TexturePreviewAnalysis> {
  const image = new Image();
  image.src = dataUrl;
  await image.decode();
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas is unavailable');
  context.drawImage(image, 0, 0);
  const sourceImage = context.getImageData(0, 0, canvas.width, canvas.height);
  const sourcePixels = new Uint8ClampedArray(sourceImage.data);
  const processedPixels = new Uint8ClampedArray(sourcePixels.length);
  const histogram = emptyHistogram();
  for (let offset = 0; offset < sourcePixels.length; offset += 4) {
    const rgba = [
      sourcePixels[offset] / 255,
      sourcePixels[offset + 1] / 255,
      sourcePixels[offset + 2] / 255,
      sourcePixels[offset + 3] / 255,
    ];
    const processed = processTexturePixel(rgba, settings);
    for (let channel = 0; channel < 4; channel += 1)
      processedPixels[offset + channel] = Math.round(processed[channel] * 255);
    histogram.r[processedPixels[offset]] += 1;
    histogram.g[processedPixels[offset + 1]] += 1;
    histogram.b[processedPixels[offset + 2]] += 1;
    histogram.a[processedPixels[offset + 3]] += 1;
  }
  context.putImageData(new ImageData(processedPixels, canvas.width, canvas.height), 0, 0);
  const processedDataUrl = canvas.toDataURL('image/png');
  const difference = new Uint8ClampedArray(sourcePixels.length);
  for (let offset = 0; offset < sourcePixels.length; offset += 4) {
    difference[offset] = Math.min(255, Math.abs(processedPixels[offset] - sourcePixels[offset]) * 4);
    difference[offset + 1] = Math.min(255, Math.abs(processedPixels[offset + 1] - sourcePixels[offset + 1]) * 4);
    difference[offset + 2] = Math.min(255, Math.abs(processedPixels[offset + 2] - sourcePixels[offset + 2]) * 4);
    difference[offset + 3] = 255;
  }
  context.putImageData(new ImageData(difference, canvas.width, canvas.height), 0, 0);
  return {
    width: canvas.width,
    height: canvas.height,
    sourceDataUrl: dataUrl,
    processedDataUrl,
    differenceDataUrl: canvas.toDataURL('image/png'),
    sourcePixels,
    processedPixels,
    histogram,
  };
}
