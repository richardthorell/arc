import { describe, expect, it } from 'vitest';

import {
  clampMaterialSidebarWidth,
  defaultMaterialSidebarWidth,
  maximumMaterialSidebarWidth,
  minimumMaterialSidebarWidth,
} from './MaterialEditor';

describe('material editor sidebar sizing', () => {
  it('starts wider for the preview and material controls', () => {
    expect(defaultMaterialSidebarWidth).toBe(640);
  });

  it('clamps resizing to the supported sidebar range', () => {
    expect(clampMaterialSidebarWidth(1600, 100)).toBe(minimumMaterialSidebarWidth);
    expect(clampMaterialSidebarWidth(1600, 900)).toBe(maximumMaterialSidebarWidth);
  });

  it('preserves the minimum graph width on narrower editor layouts', () => {
    expect(clampMaterialSidebarWidth(900, 600)).toBe(375);
  });
});
