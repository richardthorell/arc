import { describe, expect, it } from 'vitest';

import { classifyShaderExternalChange } from './shaderExternalChange';

const base = {
  localSource: 'original',
  confirmedSource: 'original',
  confirmedModifiedAt: '2026-10-05T10:00:00.000Z',
  diskSource: 'original',
  diskModifiedAt: '2026-10-05T10:00:00.000Z',
};

describe('classifyShaderExternalChange', () => {
  it('ignores a file that still matches the confirmed version', () => {
    expect(classifyShaderExternalChange(base)).toEqual({ kind: 'unchanged' });
  });

  it('reloads an externally changed file when there are no local edits', () => {
    expect(
      classifyShaderExternalChange({
        ...base,
        diskSource: 'external',
        diskModifiedAt: '2026-10-05T10:01:00.000Z',
      }),
    ).toEqual({
      kind: 'reload',
      source: 'external',
      modifiedAt: '2026-10-05T10:01:00.000Z',
    });
  });

  it('reports a conflict instead of overwriting local edits', () => {
    expect(
      classifyShaderExternalChange({
        ...base,
        localSource: 'local edit',
        diskSource: 'external edit',
        diskModifiedAt: '2026-10-05T10:01:00.000Z',
      }),
    ).toEqual({
      kind: 'conflict',
      source: 'external edit',
      modifiedAt: '2026-10-05T10:01:00.000Z',
    });
  });

  it('treats a timestamp-only external write as a disk change', () => {
    expect(
      classifyShaderExternalChange({
        ...base,
        diskModifiedAt: '2026-10-05T10:01:00.000Z',
      }),
    ).toEqual({
      kind: 'reload',
      source: 'original',
      modifiedAt: '2026-10-05T10:01:00.000Z',
    });
  });
});
