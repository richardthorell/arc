import { describe, expect, it } from 'vitest';
import { planAssetRelocation, planAssetRelocationTransaction } from './assetDependencyOperations';

describe('asset relocation path safety', () => {
  it('rejects absolute relocation paths before mutation', () => {
    expect(() => planAssetRelocation('asset-a', 'A.arc', '/Outside/A.arc')).toThrow(
      'Asset path must be relative: /Outside/A.arc',
    );
    expect(() => planAssetRelocation('asset-a', 'A.arc', 'C:\\Outside\\A.arc')).toThrow(
      'Asset path must be relative: C:\\Outside\\A.arc',
    );
  });

  it('rejects parent traversal in source and destination paths', () => {
    expect(() => planAssetRelocation('asset-a', '../A.arc', 'Moved/A.arc')).toThrow(
      'Asset path cannot traverse outside the asset root: ../A.arc',
    );
    expect(() => planAssetRelocation('asset-a', 'A.arc', 'Moved/../../A.arc')).toThrow(
      'Asset path cannot traverse outside the asset root: Moved/../../A.arc',
    );
  });

  it('applies the same safety rules to occupied paths used by bulk planning', () => {
    expect(() =>
      planAssetRelocationTransaction([{ assetId: 'asset-a', fromPath: 'A.arc', toPath: 'Moved/A.arc' }], [
        '../Outside.arc',
      ]),
    ).toThrow('Asset path cannot traverse outside the asset root: ../Outside.arc');
  });
});
