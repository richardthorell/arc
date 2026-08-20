import { describe, expect, it } from 'vitest';

import { glslSymbols } from './glslBuiltins';

const symbolsByName = new Map(glslSymbols.map((symbol) => [symbol.name, symbol]));

describe('GLSL built-in symbols', () => {
  it('keeps symbol names unique', () => {
    expect(symbolsByName.size).toBe(glslSymbols.length);
  });

  it('includes common types, functions, and stage variables', () => {
    expect(symbolsByName.get('vec3')?.kind).toBe('type');
    expect(symbolsByName.get('normalize')?.kind).toBe('function');
    expect(symbolsByName.get('texture')?.kind).toBe('function');
    expect(symbolsByName.get('gl_Position')?.kind).toBe('variable');
  });

  it('provides hover signatures for common functions', () => {
    expect(symbolsByName.get('normalize')?.signatures?.[0]?.label).toContain('normalize');
    expect(symbolsByName.get('texture')?.signatures?.length).toBeGreaterThan(0);
  });
});
