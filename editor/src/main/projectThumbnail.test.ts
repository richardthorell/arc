import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, expect, it } from 'vitest';
import { projectThumbnail, webpDimensions } from './projectThumbnail';

const roots: string[] = [];
it('checks WebP dimensions and rejects truncated or animated containers', () => {
  const bytes = Buffer.alloc(30);
  bytes.write('RIFF');
  bytes.writeUInt32LE(22, 4);
  bytes.write('WEBP', 8);
  bytes.write('VP8X', 12);
  bytes.writeUInt32LE(10, 16);
  bytes.writeUIntLE(1279, 24, 3);
  bytes.writeUIntLE(719, 27, 3);
  expect(webpDimensions(bytes)).toEqual({ width: 1280, height: 720 });
  expect(webpDimensions(bytes.subarray(0, 29))).toBeUndefined();
  bytes[20] = 2;
  expect(webpDimensions(bytes)).toBeUndefined();
});
afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});
function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-thumbnail-'));
  roots.push(root);
  const descriptor = path.join(root, 'Game.arcproject');
  const image = path.join(root, 'Preview.png');
  fs.writeFileSync(image, Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  fs.writeFileSync(descriptor, JSON.stringify({ thumbnail: 'Preview.png' }));
  return { descriptor, image };
}
it('returns a bounded raster preview without an asset database or sidecar', () => {
  expect(projectThumbnail(fixture().descriptor)).toMatch(/^data:image\/png;base64,/);
});
it('rejects non-images and oversized images', () => {
  const { descriptor, image } = fixture();
  fs.writeFileSync(image, '<svg/>');
  expect(projectThumbnail(descriptor)).toBeUndefined();
  fs.writeFileSync(image, Buffer.alloc(5 * 1024 * 1024 + 1));
  expect(projectThumbnail(descriptor)).toBeUndefined();
});
it('rejects paths outside the project and missing descriptors', () => {
  const { descriptor } = fixture();
  fs.writeFileSync(descriptor, JSON.stringify({ thumbnail: '../Preview.png' }));
  expect(projectThumbnail(descriptor)).toBeUndefined();
  expect(projectThumbnail(descriptor + '.missing')).toBeUndefined();
});
