import { describe, expect, it } from 'vitest';

import { arcUri, parseArcUri } from './arcUri';

describe('ARC URI', () => {
  it('parses resource identity, subresources, and query parameters', () => {
    const parsed = parseArcUri('arc://asset/material%20guid/thumbnail?size=128');
    expect(parsed).not.toBeNull();
    expect(parsed).toMatchObject({
      kind: 'asset',
      id: 'material guid',
      path: ['thumbnail'],
    });
    expect(parsed?.query.get('size')).toBe('128');
  });

  it('serializes deterministically with encoded path segments and sorted query keys', () => {
    expect(
      arcUri({
        kind: 'asset',
        id: 'material/guid',
        path: ['thumbnail'],
        query: { size: '128', mode: 'fit' },
      }),
    ).toBe('arc://asset/material%2Fguid/thumbnail?mode=fit&size=128');
  });

  it('rejects malformed resource kinds, IDs, duplicate query keys, and invalid encoding', () => {
    expect(parseArcUri('https://example.com')).toBeNull();
    expect(parseArcUri('arc://Asset/foo')).toBeNull();
    expect(parseArcUri('arc://asset/')).toBeNull();
    expect(parseArcUri('arc://asset/%E0%A4%A')).toBeNull();
    expect(parseArcUri('arc://asset/foo?size=64&size=128')).toBeNull();
  });
});
