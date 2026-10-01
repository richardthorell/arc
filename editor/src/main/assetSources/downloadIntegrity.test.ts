import { createHash } from 'node:crypto';

import { describe, expect, it } from 'vitest';

import { verifyRemoteDownloadIntegrity } from './downloadIntegrity';

const bytes = new TextEncoder().encode('arc remote asset');

const digest = (algorithm: 'md5' | 'sha256'): string => createHash(algorithm).update(bytes).digest('hex');

describe('remote download integrity', () => {
  it.each(['md5', 'sha256'] as const)('verifies matching %s checksums', (algorithm) => {
    const expected = digest(algorithm);
    expect(
      verifyRemoteDownloadIntegrity(
        { logicalPath: 'textures/albedo.png', checksum: { algorithm, value: expected.toUpperCase() } },
        bytes,
      ),
    ).toEqual({ verified: true, algorithm, expected, actual: expected });
  });

  it('accepts provider-prefixed checksum metadata', () => {
    const expected = digest('sha256');
    expect(
      verifyRemoteDownloadIntegrity(
        { logicalPath: 'models/source.glb', checksum: { algorithm: 'sha256', value: `sha256:${expected}` } },
        bytes,
      ).verified,
    ).toBe(true);
  });

  it('rejects checksum mismatches with the logical path and digest details', () => {
    const expected = '0'.repeat(64);
    expect(() =>
      verifyRemoteDownloadIntegrity(
        { logicalPath: 'models/source.glb', checksum: { algorithm: 'sha256', value: expected } },
        bytes,
      ),
    ).toThrow("Remote asset integrity check failed for 'models/source.glb': SHA256 checksum mismatch");
  });

  it('rejects malformed provider checksum metadata before hashing', () => {
    expect(() =>
      verifyRemoteDownloadIntegrity(
        { logicalPath: 'textures/albedo.png', checksum: { algorithm: 'md5', value: 'not-a-digest' } },
        bytes,
      ),
    ).toThrow('invalid MD5 digest');
  });

  it('keeps files without checksum metadata explicitly unverified', () => {
    expect(verifyRemoteDownloadIntegrity({ logicalPath: 'textures/albedo.png' }, bytes)).toEqual({ verified: false });
  });
});
