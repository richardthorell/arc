import { describe, expect, it } from 'vitest';

import { arcAssetDragMime, parseArcAssetDragPayload, readArcAssetDragPayload } from './assetDragPayload';

describe('assetDragPayload', () => {
  it('decodes the Content Browser JSON payload', () => {
    expect(
      parseArcAssetDragPayload(
        JSON.stringify({ guid: 'material-guid', type: 'material', pathHint: 'Content/Materials/Hero.arcmat' }),
      ),
    ).toEqual({ guid: 'material-guid', type: 'material', pathHint: 'Content/Materials/Hero.arcmat' });
  });

  it('keeps compatibility with legacy plain-path payloads', () => {
    expect(parseArcAssetDragPayload('Content/Textures/Albedo.png')).toEqual({
      guid: '',
      type: '',
      pathHint: 'Content/Textures/Albedo.png',
    });
  });

  it('falls back to legacy environment drag data', () => {
    const transfer = {
      getData: (type: string) =>
        type === arcAssetDragMime ? '' : type === 'application/x-arc-environment' ? 'Content/Sky.hdr' : '',
    };
    expect(readArcAssetDragPayload(transfer)).toEqual({ guid: '', type: 'environment', pathHint: 'Content/Sky.hdr' });
  });
});
