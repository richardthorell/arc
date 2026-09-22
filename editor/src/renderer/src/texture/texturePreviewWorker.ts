import { processTexturePixel } from './texturePixelProcessing';
import type { TextureHistogram } from './texturePreviewProcessing';
import type { TextureSettingsSnapshot } from './textureSettings';

self.onmessage = (event: MessageEvent<{ samples: Uint8ClampedArray; settings: TextureSettingsSnapshot }>) => {
  const { samples, settings } = event.data;
  const histogram: TextureHistogram = {
    r: Array(256).fill(0),
    g: Array(256).fill(0),
    b: Array(256).fill(0),
    a: Array(256).fill(0),
  };
  const channels = [histogram.r, histogram.g, histogram.b, histogram.a];
  for (let offset = 0; offset < samples.length; offset += 4) {
    const processed = processTexturePixel(
      [samples[offset] / 255, samples[offset + 1] / 255, samples[offset + 2] / 255, samples[offset + 3] / 255],
      settings,
    );
    for (let channel = 0; channel < 4; channel += 1) channels[channel][Math.round(processed[channel] * 255)] += 1;
  }
  self.postMessage(histogram);
};
