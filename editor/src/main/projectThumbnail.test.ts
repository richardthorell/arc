import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, expect, it } from 'vitest';
import { projectThumbnail } from './projectThumbnail';

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});
function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-thumbnail-'));
  roots.push(root);
  const descriptor = path.join(root, 'Game.arcproject');
  const image = path.join(root, 'Preview.png');
  fs.writeFileSync(image, Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  fs.writeFileSync(image + '.arcmeta', JSON.stringify({ guid: 'image' }));
  fs.writeFileSync(
    descriptor,
    JSON.stringify({ thumbnail: { guid: 'image', expectedType: 'texture', pathHint: 'Preview.png' } }),
  );
  return { descriptor, image };
}
it('returns a bounded raster preview for the matching project asset', () => {
  expect(projectThumbnail(fixture().descriptor)).toMatch(/^data:image\/png;base64,/);
});
it('rejects mismatched identities, non-images and oversized images', () => {
  const { descriptor, image } = fixture();
  fs.writeFileSync(image + '.arcmeta', JSON.stringify({ guid: 'other' }));
  expect(projectThumbnail(descriptor)).toBeUndefined();
  fs.writeFileSync(image + '.arcmeta', JSON.stringify({ guid: 'image' }));
  fs.writeFileSync(image, '<svg/>');
  expect(projectThumbnail(descriptor)).toBeUndefined();
  fs.writeFileSync(image, Buffer.alloc(1024 * 1024 + 1));
  expect(projectThumbnail(descriptor)).toBeUndefined();
});
it('rejects paths outside the project and missing descriptors', () => {
  const { descriptor } = fixture();
  fs.writeFileSync(
    descriptor,
    JSON.stringify({ thumbnail: { guid: 'image', expectedType: 'texture', pathHint: '../Preview.png' } }),
  );
  expect(projectThumbnail(descriptor)).toBeUndefined();
  expect(projectThumbnail(descriptor + '.missing')).toBeUndefined();
});
