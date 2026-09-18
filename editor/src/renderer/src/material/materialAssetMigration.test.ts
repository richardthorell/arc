import { describe, expect, it } from 'vitest';

import { currentMaterialAuthoringVersion, upgradeMaterialAsset } from './materialAssetMigration';
import { createDefaultMaterialGraph, materialGraphFromAsset } from './materialGraphTypes';

describe('material asset migration', () => {
  it('leaves current compiled-only materials unchanged', () => {
    const asset = { version: 4, name: 'Current', graph: createDefaultMaterialGraph() };
    const result = upgradeMaterialAsset(asset);

    expect(result.upgraded).toBe(false);
    expect(result.sourceVersion).toBe(currentMaterialAuthoringVersion);
    expect(result.asset).toBe(asset);
  });

  it('upgrades a legacy descriptor material to the current graph schema without losing authored values', () => {
    const result = upgradeMaterialAsset({
      version: 3,
      name: 'Legacy',
      shader: 'arc/default_phong',
      domain: 'surface',
      blendMode: 'opaque',
      shadingModel: 'standard',
      surface: {
        baseColor: { r: 0.2, g: 0.4, b: 0.8, a: 0.75 },
        metallic: 0.7,
        roughness: 0.25,
        normalScale: 0.6,
        aoStrength: 0.9,
        emissive: { r: 0.1, g: 0.2, b: 0.3 },
        emissiveStrength: 2,
        alphaCutoff: 0.4,
      },
      textures: {
        normal: 'Textures/normal.png',
        height: 'Textures/height.png',
      },
      advanced: {
        clearCoat: 0.5,
        anisotropy: 0.35,
        parallaxHeightScale: 0.04,
      },
      graph: null,
      futureEditorMetadata: { keep: true },
    });

    expect(result.upgraded).toBe(true);
    expect(result.sourceVersion).toBe(3);
    expect(result.asset.version).toBe(4);
    expect(result.asset).not.toHaveProperty('shader');
    expect(result.asset).not.toHaveProperty('surface');
    expect(result.asset).not.toHaveProperty('textures');
    expect(result.asset).not.toHaveProperty('advanced');
    expect(result.asset.futureEditorMetadata).toEqual({ keep: true });
    expect(result.asset.migrationMetadata).toEqual({
      legacyShader: 'arc/default_phong',
      legacyHeightTexture: 'Textures/height.png',
      legacyParallaxHeightScale: 0.04,
    });

    const graph = materialGraphFromAsset(result.asset);
    expect(graph.nodes.find((node) => node.id === 'legacy-base-color')?.values.value).toEqual([0.2, 0.4, 0.8]);
    expect(graph.nodes.find((node) => node.id === 'legacy-metallic')?.values.value).toBe(0.7);
    expect(graph.nodes.find((node) => node.id === 'legacy-roughness')?.values.value).toBe(0.25);
    expect(graph.nodes.find((node) => node.id === 'legacy-emissive')?.values.value).toEqual([0.2, 0.4, 0.6]);
    expect(graph.nodes.find((node) => node.id === 'legacy-normal-map')?.values.strength).toBe(0.6);
    expect(graph.connections.some((connection) => connection.to.pin === 'clearCoat')).toBe(true);
    expect(graph.connections.some((connection) => connection.to.pin === 'anisotropy')).toBe(true);
  });

  it('keeps an existing legacy graph while upgrading only its authoring envelope', () => {
    const graph = createDefaultMaterialGraph();
    const result = upgradeMaterialAsset({
      version: 3,
      name: 'Graph Material',
      graph,
      editorMetadata: { custom: true },
    });

    expect(result.upgraded).toBe(true);
    expect(result.asset.version).toBe(4);
    expect(materialGraphFromAsset(result.asset)).toEqual(graph);
    expect(result.asset.editorMetadata).toEqual({ custom: true });
  });

  it('does not silently downgrade future material schemas', () => {
    expect(() => upgradeMaterialAsset({ version: 5, graph: createDefaultMaterialGraph() })).toThrow(
      'Unsupported material authoring schema v5',
    );
  });
});
