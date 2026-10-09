import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  deserializeMaterialInstanceAsset,
  materialFunctionSlotParameterId,
  materialParameterId,
} from './materialInstancePersistence';

const assetsRoot = path.resolve(process.cwd(), '..', 'assets');
const floorPath = path.join(assetsRoot, 'materials', 'floor.arcmatinst');
const checkerTexturePath = path.join(assetsRoot, 'textures', 'editor', 'default_checker_floor.png');

describe('built-in Floor Material Instance', () => {
  it('specializes Standard Lit with the reusable Checker function', () => {
    const floor = deserializeMaterialInstanceAsset(fs.readFileSync(floorPath, 'utf8'));
    expect(floor).not.toBeNull();
    expect(floor).toMatchObject({
      version: 1,
      name: 'Floor',
      parent: {
        guid: 'b36d1554-ddb0-4692-8a2c-a7dde15e0456',
        pathHint: 'materials/standard_lit.arcmat',
      },
    });

    expect(floor?.parameterOverrides).toEqual([
      { parameterId: materialParameterId('metallic'), value: 0 },
      { parameterId: materialParameterId('roughness'), value: 0.8 },
    ]);

    const checker = floor?.functionOverrides[0];
    expect(checker).toMatchObject({
      slotId: 'base-color-source',
      function: {
        guid: '76433716-d257-4cc4-974a-921f393a9602',
        pathHint: 'material_functions/checker.arcmatfn',
      },
      inputOverrides: [
        { pinId: 'colorA', value: [0.78, 0.78, 0.78] },
        { pinId: 'colorB', value: [0.32, 0.32, 0.32] },
        { pinId: 'cellSize', value: 1 },
      ],
    });
    expect(floor?.functionOverrides).toHaveLength(1);

    const checkerGuid = checker?.function.guid ?? '';
    expect(materialFunctionSlotParameterId('base-color-source', checkerGuid, 'colorA')).toBe('4654449916007669735');
    expect(materialFunctionSlotParameterId('base-color-source', checkerGuid, 'colorB')).toBe('4654451015519297946');
    expect(materialFunctionSlotParameterId('base-color-source', checkerGuid, 'cellSize')).toBe('4459348377896743872');
  });

  it('is a first-class built-in Material Instance and no longer depends on the checker PNG', () => {
    const metadata = JSON.parse(fs.readFileSync(`${floorPath}.arcmeta`, 'utf8')) as {
      type: string;
      importer: string;
      guid: string;
      title: string;
    };
    expect(metadata).toMatchObject({
      type: 'a7ca55e7-0000-0001-0000-00000000000e',
      importer: 'a7ca55e7-0000-0002-0000-00000000000e',
      guid: '6a8c7339-3a9f-49d5-9d4f-1b5a73d02a91',
      title: 'Floor',
    });
    expect(fs.existsSync(checkerTexturePath)).toBe(false);
    expect(fs.existsSync(`${checkerTexturePath}.arcmeta`)).toBe(false);
  });
});
