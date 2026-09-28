import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const migratedTerrainSurfaces = [
  'CreateTerrainDialog.tsx',
  'TerrainStackPanel.tsx',
  'TerrainToolsPanel.tsx',
  'TerrainViewportOverlay.tsx',
] as const;

const nativeControlPattern = /<(?:button|input|select|textarea)\b/i;

describe('terrain shared control migration', () => {
  it.each(migratedTerrainSurfaces)('%s stays on shared Ui controls', (filename) => {
    const source = readFileSync(fileURLToPath(new URL(filename, import.meta.url)), 'utf8');

    expect(source).not.toMatch(nativeControlPattern);
  });
});
