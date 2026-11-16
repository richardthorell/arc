import fs from 'node:fs';
import path from 'node:path';

/** Read bounded WebP dimensions without an asset database or platform image codec. */
export function webpDimensions(bytes: Buffer): { width: number; height: number } | undefined {
  if (
    bytes.length < 20 ||
    bytes.toString('ascii', 0, 4) !== 'RIFF' ||
    bytes.toString('ascii', 8, 12) !== 'WEBP' ||
    bytes.readUInt32LE(4) + 8 !== bytes.length
  )
    return;
  for (let offset = 12; offset + 8 <= bytes.length;) {
    const kind = bytes.toString('ascii', offset, offset + 4);
    const length = bytes.readUInt32LE(offset + 4);
    const start = offset + 8;
    if (start + length > bytes.length) return;
    if (kind === 'VP8X' && length >= 10) {
      if (bytes[start] & 2) return; // Project thumbnails are static.
      return { width: 1 + bytes.readUIntLE(start + 4, 3), height: 1 + bytes.readUIntLE(start + 7, 3) };
    }
    if (kind === 'VP8 ' && length >= 10 && bytes.subarray(start + 3, start + 6).equals(Buffer.from([0x9d, 1, 0x2a])))
      return { width: bytes.readUInt16LE(start + 6) & 0x3fff, height: bytes.readUInt16LE(start + 8) & 0x3fff };
    if (kind === 'VP8L' && length >= 5 && bytes[start] === 0x2f) {
      const bits = bytes.readUInt32LE(start + 1);
      return { width: 1 + (bits & 0x3fff), height: 1 + ((bits >>> 14) & 0x3fff) };
    }
    offset = start + length + (length & 1);
  }
}

/** Bounded, project-contained raster preview. Never expose native file URLs. */
export function projectThumbnail(descriptorPath: string): string | undefined {
  try {
    if (fs.statSync(descriptorPath).size > 1024 * 1024) return;
    const descriptor = JSON.parse(fs.readFileSync(descriptorPath, 'utf8'));
    const reference = descriptor.thumbnail;
    if (typeof reference !== 'string' || path.isAbsolute(reference)) return;
    const root = fs.realpathSync(path.dirname(descriptorPath));
    const file = fs.realpathSync(path.resolve(root, reference));
    const relative = path.relative(root, file);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) return;
    if (!fs.statSync(file).isFile() || fs.statSync(file).size > 5 * 1024 * 1024) return;
    const data = fs.readFileSync(file);
    const extension = path.extname(file).toLowerCase();
    const type =
      extension === '.png' && data.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
        ? 'png'
        : ['.jpg', '.jpeg'].includes(extension) && data[0] === 255 && data[1] === 216 && data[2] === 255
          ? 'jpeg'
          : extension === '.webp' && data.toString('ascii', 0, 4) === 'RIFF' && data.toString('ascii', 8, 12) === 'WEBP'
            ? 'webp'
            : undefined;
    return type ? `data:image/${type};base64,${data.toString('base64')}` : undefined;
  } catch {
    return undefined;
  }
}
