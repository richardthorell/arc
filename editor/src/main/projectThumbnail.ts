import fs from 'node:fs';
import path from 'node:path';

/** Bounded, project-contained raster preview. Never expose native file URLs. */
export function projectThumbnail(descriptorPath: string): string | undefined {
  try {
    if (fs.statSync(descriptorPath).size > 1024 * 1024) return;
    const descriptor = JSON.parse(fs.readFileSync(descriptorPath, 'utf8'));
    const reference = descriptor.thumbnail;
    if (
      !reference?.guid ||
      reference.expectedType !== 'texture' ||
      typeof reference.pathHint !== 'string' ||
      path.isAbsolute(reference.pathHint)
    )
      return;
    const root = fs.realpathSync(path.dirname(descriptorPath));
    const file = fs.realpathSync(path.resolve(root, reference.pathHint));
    const relative = path.relative(root, file);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) return;
    const metadata = file + '.arcmeta';
    if (fs.statSync(metadata).size > 65536 || JSON.parse(fs.readFileSync(metadata, 'utf8')).guid !== reference.guid)
      return;
    if (!fs.statSync(file).isFile() || fs.statSync(file).size > 1024 * 1024) return;
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
