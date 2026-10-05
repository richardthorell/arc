import { describe, expect, it } from 'vitest';

import { resolveShaderIncludeTarget } from './shaderIncludeNavigation';

describe('shader include navigation', () => {
  it('resolves relative includes from the active shader directory', () => {
    expect(resolveShaderIncludeTarget('../common/lighting.glsl', 'shaders/pbr/main.frag')).toEqual({
      path: 'shaders/common/lighting.glsl',
    });
  });

  it('normalizes Windows separators before resolving', () => {
    expect(resolveShaderIncludeTarget('include\\common.glsl', 'shaders\\main.frag')).toEqual({
      path: 'shaders/include/common.glsl',
    });
  });

  it('preserves project-root and drive-qualified include targets', () => {
    expect(resolveShaderIncludeTarget('/shaders/common.glsl', 'shaders/main.frag')).toEqual({
      path: '/shaders/common.glsl',
    });
    expect(resolveShaderIncludeTarget('C:\\arc\\shaders\\common.glsl', 'shaders/main.frag')).toEqual({
      path: 'C:/arc/shaders/common.glsl',
    });
  });

  it('returns a normalized include when no active document is available', () => {
    expect(resolveShaderIncludeTarget('./common.glsl')).toEqual({ path: 'common.glsl' });
  });

  it('rejects empty include targets', () => {
    expect(resolveShaderIncludeTarget('   ', 'shaders/main.frag')).toBeUndefined();
  });
});
