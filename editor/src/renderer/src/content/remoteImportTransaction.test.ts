import { describe, expect, it } from 'vitest';
import {
  beginRemoteAssetImport,
  beginRemoteImportDownload,
  cancelRemoteImport,
  commitRemoteImport,
  createRemoteImportTransaction,
  publishRemoteImportedAsset,
  remoteImportCleanup,
  stageRemoteImportPath,
} from './remoteImportTransaction';

describe('remote import transaction', () => {
  it('preserves staged work for cleanup when download is cancelled', () => {
    let transaction = beginRemoteImportDownload(createRemoteImportTransaction());
    transaction = stageRemoteImportPath(transaction, 'cache/albedo.png');
    transaction = stageRemoteImportPath(transaction, 'cache/normal.png');
    transaction = cancelRemoteImport(transaction);

    expect(transaction.state).toBe('cancelled');
    expect(remoteImportCleanup(transaction)).toEqual({
      stagedPaths: ['cache/albedo.png', 'cache/normal.png'],
      publishedAssetIds: [],
    });
  });

  it('preserves partially published assets for rollback when import is cancelled', () => {
    let transaction = beginRemoteImportDownload(createRemoteImportTransaction());
    transaction = stageRemoteImportPath(transaction, 'cache/source.glb');
    transaction = beginRemoteAssetImport(transaction);
    transaction = publishRemoteImportedAsset(transaction, 'asset-mesh');
    transaction = cancelRemoteImport(transaction);

    expect(remoteImportCleanup(transaction)).toEqual({
      stagedPaths: ['cache/source.glb'],
      publishedAssetIds: ['asset-mesh'],
    });
  });

  it('deduplicates cleanup ownership and seals terminal transactions', () => {
    let transaction = beginRemoteImportDownload(createRemoteImportTransaction());
    transaction = stageRemoteImportPath(transaction, 'cache/source.glb');
    transaction = stageRemoteImportPath(transaction, 'cache/source.glb');
    transaction = beginRemoteAssetImport(transaction);
    transaction = publishRemoteImportedAsset(transaction, 'asset-mesh');
    transaction = publishRemoteImportedAsset(transaction, 'asset-mesh');
    transaction = commitRemoteImport(transaction);

    expect(remoteImportCleanup(transaction)).toEqual({
      stagedPaths: ['cache/source.glb'],
      publishedAssetIds: ['asset-mesh'],
    });
    expect(() => cancelRemoteImport(transaction)).toThrow('already committed');
  });

  it('rejects publishing before the import stage', () => {
    const transaction = beginRemoteImportDownload(createRemoteImportTransaction());
    expect(() => publishRemoteImportedAsset(transaction, 'asset-mesh')).toThrow('only be published while importing');
  });
});
