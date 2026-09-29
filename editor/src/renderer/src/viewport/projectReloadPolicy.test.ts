import { describe, expect, it } from 'vitest';
import { classifyProjectReload } from './projectReloadPolicy';

describe('project reload policy', () => {
  it('keeps compatible additive and rename-only changes hot reloadable', () => {
    expect(classifyProjectReload(['additive-schema', 'rename-only'])).toEqual({
      action: 'hot-reload',
      reasons: ['additive-schema', 'rename-only'],
    });
  });

  it('requires a clean Play-session restart for field wire-kind changes', () => {
    expect(classifyProjectReload(['field-wire-kind'])).toEqual({
      action: 'restart-play-session',
      reasons: ['field-wire-kind'],
    });
  });

  it.each(['removed-component', 'schema-downgrade', 'identity-change', 'abi-change'] as const)(
    'requires a native-host restart for %s',
    (change) => {
      expect(classifyProjectReload([change])).toEqual({
        action: 'restart-native-host',
        reasons: [change],
      });
    },
  );

  it('lets the most restrictive change win while preserving all reasons', () => {
    expect(classifyProjectReload(['additive-schema', 'field-wire-kind', 'abi-change'])).toEqual({
      action: 'restart-native-host',
      reasons: ['additive-schema', 'field-wire-kind', 'abi-change'],
    });
  });

  it('deduplicates restart reasons deterministically', () => {
    expect(classifyProjectReload(['field-wire-kind', 'field-wire-kind'])).toEqual({
      action: 'restart-play-session',
      reasons: ['field-wire-kind'],
    });
  });
});
