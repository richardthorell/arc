import { createHash } from 'node:crypto';

import type { ArcAssetDownloadFile } from '../../common/assetSourceTypes';

export type RemoteDownloadIntegrityResult = Readonly<{
  verified: boolean;
  algorithm?: 'md5' | 'sha256';
  expected?: string;
  actual?: string;
}>;

const normalizeDigest = (value: string): string => value.trim().toLocaleLowerCase().replace(/^sha256:/, '').replace(/^md5:/, '');

const digestLength: Readonly<Record<'md5' | 'sha256', number>> = {
  md5: 32,
  sha256: 64,
};

const validateExpectedDigest = (algorithm: 'md5' | 'sha256', value: string): string => {
  const normalized = normalizeDigest(value);
  if (normalized.length !== digestLength[algorithm] || !/^[0-9a-f]+$/.test(normalized)) {
    throw new Error(`Remote asset checksum metadata contains an invalid ${algorithm.toUpperCase()} digest`);
  }
  return normalized;
};

/**
 * Verifies downloaded bytes against provider checksum metadata before they can
 * advance to ARC's importer/cooker. Files without provider checksum metadata
 * remain explicitly unverified rather than being treated as verified.
 */
export const verifyRemoteDownloadIntegrity = (
  file: Pick<ArcAssetDownloadFile, 'logicalPath' | 'checksum'>,
  bytes: Uint8Array,
): RemoteDownloadIntegrityResult => {
  if (!file.checksum) return { verified: false };

  const { algorithm } = file.checksum;
  const expected = validateExpectedDigest(algorithm, file.checksum.value);
  const actual = createHash(algorithm).update(bytes).digest('hex');

  if (actual !== expected) {
    const path = file.logicalPath || '<unknown>';
    throw new Error(
      `Remote asset integrity check failed for '${path}': ${algorithm.toUpperCase()} checksum mismatch (expected ${expected}, got ${actual})`,
    );
  }

  return { verified: true, algorithm, expected, actual };
};
