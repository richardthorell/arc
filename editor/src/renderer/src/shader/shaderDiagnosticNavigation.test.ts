import { describe, expect, it } from 'vitest';

import {
  resolveShaderDiagnosticTarget,
  routeShaderDiagnostic,
  shaderDiagnosticTargetKey,
} from './shaderDiagnosticNavigation';

describe('shader diagnostic navigation', () => {
  it('prefers the compiler path and normalizes separators', () => {
    expect(
      resolveShaderDiagnosticTarget(
        { path: 'shaders\\lighting\\common.glsl', line: 42, column: 7 },
        'shaders/main.frag',
      ),
    ).toEqual({ path: 'shaders/lighting/common.glsl', line: 42, column: 7 });
  });

  it('falls back to the active document when the compiler omits a path', () => {
    expect(resolveShaderDiagnosticTarget({ line: 9 }, ' shaders/main.frag ')).toEqual({
      path: 'shaders/main.frag',
      line: 9,
      column: 1,
    });
  });

  it('rejects diagnostics without a valid source line or path', () => {
    expect(resolveShaderDiagnosticTarget({ path: 'shader.frag', line: 0 })).toBeUndefined();
    expect(resolveShaderDiagnosticTarget({ path: 'shader.frag', line: -2 })).toBeUndefined();
    expect(resolveShaderDiagnosticTarget({ line: 4 })).toBeUndefined();
  });

  it('normalizes invalid columns to the first column', () => {
    expect(resolveShaderDiagnosticTarget({ path: 'shader.frag', line: 3, column: 0 })).toEqual({
      path: 'shader.frag',
      line: 3,
      column: 1,
    });
  });

  it('routes diagnostics for the active source to the current editor', () => {
    expect(routeShaderDiagnostic({ path: '.\\shaders\\main.frag', line: 14, column: 2 }, 'shaders/main.frag')).toEqual({
      kind: 'active-document',
      target: { path: './shaders/main.frag', line: 14, column: 2 },
    });
  });

  it('routes include diagnostics to an external document instead of the active editor', () => {
    expect(routeShaderDiagnostic({ path: 'shaders/include/lighting.glsl', line: 27 }, 'shaders/main.frag')).toEqual({
      kind: 'external-document',
      target: { path: 'shaders/include/lighting.glsl', line: 27, column: 1 },
    });
  });

  it('treats pathless diagnostics as belonging to the active document', () => {
    expect(routeShaderDiagnostic({ line: 8 }, 'shaders/main.frag')).toEqual({
      kind: 'active-document',
      target: { path: 'shaders/main.frag', line: 8, column: 1 },
    });
  });

  it('builds stable location keys for editor selection/history', () => {
    expect(shaderDiagnosticTargetKey({ path: 'include/common.glsl', line: 12, column: 5 })).toBe(
      'include/common.glsl:12:5',
    );
  });
});
