import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const terrainDirectory = fileURLToPath(new URL('.', import.meta.url));
const terrainSurfaces = readdirSync(terrainDirectory)
  .filter((filename) => filename.endsWith('.tsx'))
  .sort();

const nativeControlPattern = /<(?:button|input|select|textarea)\b/i;

describe('terrain shared control migration', () => {
  it('keeps the terrain surface inventory covered', () => {
    expect(terrainSurfaces.length).toBeGreaterThan(0);
  });

  it.each(terrainSurfaces)('%s stays on shared Ui controls', (filename) => {
    const source = readFileSync(new URL(filename, import.meta.url), 'utf8');

    expect(source).not.toMatch(nativeControlPattern);
  });
});
