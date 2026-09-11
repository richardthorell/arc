import { describe, expect, it } from 'vitest';

import { assetPresentationIcon, assetPresentationKind, assetPresentationLabel } from './assetPresentation';

describe('Flow asset presentation', () => {
  it('recognizes .arcflow files even when the native kind is unknown', () => {
    const asset = { kind: 'unknown' as const, path: 'Content/Logic/Player.arcflow' };

    expect(assetPresentationKind(asset)).toBe('flow');
    expect(assetPresentationLabel(asset)).toBe('Flow Graph');
    expect(assetPresentationIcon(asset)).toBe('script');
  });
});
