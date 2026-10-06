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
        clearCoat: 'Textures/clear_coat.png',
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
    expect(graph.connections.filter((connection) => connection.to.pin === 'clearCoat')).toHaveLength(1);
    expect(graph.nodes.find((node) => node.id === 'legacy-clear-coat-texture-texture')?.values.texture).toBe(
      'Textures/clear_coat.png',
    );
    expect(graph.connections.some((connection) => connection.to.pin === 'anisotropy')).toBe(true);
  });

  it('renames the canonical legacy Default Phong identity while preserving its compatibility shader reference', () => {
    const result = upgradeMaterialAsset({
      version: 3,
      name: 'Default Phong',
      shader: 'arc/default_phong',
      domain: 'surface',
      blendMode: 'opaque',
      shadingModel: 'standard',
      graph: null,
    });

    expect(result.upgraded).toBe(true);
    expect(result.asset.name).toBe('Standard Lit');
    expect(result.asset).not.toHaveProperty('shader');
    expect(result.asset.migrationMetadata).toMatchObject({
      legacyShader: 'arc/default_phong',
    });
  });

  it('migrates identical legacy inputs deterministically and is idempotent after upgrade', () => {
    const legacy = {
      version: 3,
      name: 'Deterministic Legacy',
      domain: 'surface',
      blendMode: 'masked',
      surface: {
        baseColor: { r: 0.15, g: 0.35, b: 0.55, a: 0.8 },
        metallic: 0.25,
        roughness: 0.45,
        alphaCutoff: 0.3,
      },
      textures: {
        baseColor: 'Textures/base.png',
        metallicRoughness: 'Textures/surface.png',
        normal: 'Textures/normal.png',
      },
      graph: null,
    };

    const first = upgradeMaterialAsset(structuredClone(legacy));
    const second = upgradeMaterialAsset(structuredClone(legacy));

    expect(first.upgraded).toBe(true);
    expect(second.upgraded).toBe(true);
    expect(first.asset).toEqual(second.asset);
    expect(materialGraphFromAsset(first.asset)).toEqual(materialGraphFromAsset(second.asset));

    const repeated = upgradeMaterialAsset(first.asset);
    expect(repeated.upgraded).toBe(false);
    expect(repeated.sourceVersion).toBe(currentMaterialAuthoringVersion);
    expect(repeated.asset).toBe(first.asset);
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
