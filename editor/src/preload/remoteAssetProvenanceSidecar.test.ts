import { describe, expect, it } from 'vitest';

import { createRemoteImportProvenanceSidecar, serializeRemoteImportProvenanceSidecar } from './remoteAssetProvenance';
import { parseRemoteImportProvenanceSidecar } from './remoteAssetProvenanceSidecar';

const sidecar = createRemoteImportProvenanceSidecar(
  {
    sourceId: 'polyhaven',
    sourceAssetId: 'rock_01',
    importedAt: '2026-10-04T20:00:00.000Z',
    license: 'CC0',
    sourceUrl: 'https://polyhaven.com/a/rock_01',
    sourceRevision: 'revision-7',
    sourceHash: 'sha256:abc123',
    recipe: {
      version: 1,
      logicalPaths: ['4k/rock.glb'],
      options: { generateTangents: true, lodBias: 1 },
    },
  },
  ['Content/External/polyhaven/rock_01/rock.glb'],
  ['asset-guid-1'],
);

describe('remote asset provenance sidecar parsing', () => {
  it('round-trips the authoritative serialized sidecar without losing the reimport recipe', () => {
    const parsed = parseRemoteImportProvenanceSidecar(serializeRemoteImportProvenanceSidecar(sidecar));

    expect(parsed).toEqual(sidecar);
    expect(parsed?.provenance.recipe).toEqual(sidecar.provenance.recipe);
    expect(parsed?.provenance.sourceHash).toBe('sha256:abc123');
  });

  it('rejects malformed or unsupported sidecars instead of partially trusting provenance', () => {
    expect(parseRemoteImportProvenanceSidecar('{not-json')).toBeNull();
    expect(parseRemoteImportProvenanceSidecar(JSON.stringify({ ...sidecar, version: 2 }))).toBeNull();
    expect(
      parseRemoteImportProvenanceSidecar(
        JSON.stringify({
          ...sidecar,
          provenance: { ...sidecar.provenance, recipe: { version: 1, logicalPaths: '4k/rock.glb' } },
        }),
      ),
    ).toBeNull();
    expect(
      parseRemoteImportProvenanceSidecar(
        JSON.stringify({
          ...sidecar,
          provenance: { ...sidecar.provenance, recipe: { ...sidecar.provenance.recipe, options: { lodBias: [] } } },
        }),
      ),
    ).toBeNull();
  });

  it('returns detached arrays and option records suitable for safe reimport planning', () => {
    const parsed = parseRemoteImportProvenanceSidecar(serializeRemoteImportProvenanceSidecar(sidecar));
    expect(parsed).not.toBeNull();

    parsed!.importedFiles.push('mutated');
    parsed!.importedAssetIds.push('mutated');
    parsed!.provenance.recipe!.logicalPaths.push('mutated');
    parsed!.provenance.recipe!.options!.lodBias = 9;

    expect(sidecar.importedFiles).toEqual(['Content/External/polyhaven/rock_01/rock.glb']);
    expect(sidecar.importedAssetIds).toEqual(['asset-guid-1']);
    expect(sidecar.provenance.recipe?.logicalPaths).toEqual(['4k/rock.glb']);
    expect(sidecar.provenance.recipe?.options?.lodBias).toBe(1);
  });
});
