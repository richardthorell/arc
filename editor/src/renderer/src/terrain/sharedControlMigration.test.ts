import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

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
      const source = readSurface(fileName);

      expect(source, `${fileName} should not use native buttons`).not.toMatch(/<button\b/);
      expect(source, `${fileName} should not use native selects`).not.toMatch(/<select\b/);
      expect(source, `${fileName} should not use native text or numeric inputs`).not.toMatch(
        /<input\b(?![^>]*\btype=["'](?:file|hidden)["'])/,
      );
      expect(source, `${fileName} should not use native textareas`).not.toMatch(/<textarea\b/);
    }
  });

  it('keeps the terrain tools on ARC shared numeric controls', () => {
    const toolsSource = readSurface('TerrainToolsPanel.tsx');

    expect(toolsSource).toContain('UiSlider');
    expect(toolsSource).toContain('UiNumericInput');
    expect(toolsSource).toContain('UiButton');
  });
});
