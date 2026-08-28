import { describe, expect, it } from 'vitest';

import { materialNodeWidth } from './MaterialGraphEditor';

describe('material graph layout', () => {
  it('keeps simple nodes compact and widens controls that need more editing space', () => {
    expect(materialNodeWidth('constant')).toBe(214);
    expect(materialNodeWidth('vector2')).toBeGreaterThan(materialNodeWidth('constant'));
    expect(materialNodeWidth('vector3')).toBeGreaterThan(materialNodeWidth('vector2'));
    expect(materialNodeWidth('vector4')).toBeGreaterThan(materialNodeWidth('vector3'));
    expect(materialNodeWidth('textureSample')).toBeGreaterThan(materialNodeWidth('constant'));
    expect(materialNodeWidth('colorRgba')).toBeGreaterThan(materialNodeWidth('textureSample'));
    expect(materialNodeWidth('output')).toBeGreaterThan(materialNodeWidth('constant'));
  });
});
