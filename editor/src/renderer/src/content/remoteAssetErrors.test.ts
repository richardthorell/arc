import { describe, expect, it } from 'vitest';

import { remoteAssetFailureMessage } from './remoteAssetErrors';

describe('remoteAssetFailureMessage', () => {
  it('preserves useful provider context', () => {
    expect(remoteAssetFailureMessage('manifest', new Error('Variant metadata unavailable'))).toBe(
      'Unable to load download variants. Variant metadata unavailable',
    );
  });

  it('turns integrity failures into an actionable retry message', () => {
    expect(remoteAssetFailureMessage('import', new Error('Checksum mismatch for albedo.png'))).toBe(
      'Unable to import this asset. The downloaded file failed integrity validation. Try the download again.',
    );
  });

  it('distinguishes provider connectivity failures', () => {
    expect(remoteAssetFailureMessage('search', new Error('Network request failed'))).toBe(
      'Unable to browse remote assets. The asset provider could not be reached. Check your connection and try again.',
    );
    expect(remoteAssetFailureMessage('import', 'request timed out')).toBe(
      'Unable to import this asset. The asset provider timed out. Check your connection and try again.',
    );
  });

  it('handles missing or non-error failures without exposing transport values', () => {
    expect(remoteAssetFailureMessage('search', undefined)).toBe(
      'Unable to browse remote assets. Try again, or check the asset provider connection.',
    );
    expect(remoteAssetFailureMessage('import', { code: 500 })).toBe(
      'Unable to import this asset. Try again, or check the asset provider connection.',
    );
  });
});
