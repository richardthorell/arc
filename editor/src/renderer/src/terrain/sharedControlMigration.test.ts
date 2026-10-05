import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { findNativeControlViolations, formatNativeControlViolations } from '../testing/sharedControlMigration';

const terrainSurfaces = [
  'CreateTerrainDialog.tsx',
  'TerrainStackPanel.tsx',
  'TerrainToolsPanel.tsx',
  'TerrainViewportOverlay.tsx',
] as const;

const readSurface = (fileName: (typeof terrainSurfaces)[number]) =>
  readFileSync(new URL(`./${fileName}`, import.meta.url), 'utf8');

describe('terrain shared control migration', () => {
  it('keeps current terrain editor surfaces on shared controls', () => {
    for (const fileName of terrainSurfaces) {
      const violations = findNativeControlViolations(readSurface(fileName));

      expect(violations, formatNativeControlViolations(fileName, violations)).toEqual([]);
    }
  });

  it('keeps the terrain tools on ARC shared numeric controls', () => {
    const toolsSource = readSurface('TerrainToolsPanel.tsx');

    expect(toolsSource).toContain('UiSlider');
    expect(toolsSource).toContain('UiNumericInput');
    expect(toolsSource).toContain('UiButton');
  });
});
