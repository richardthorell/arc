import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialFunctionAssetJson } from './materialGraphTypes';

const builtInFunctionRoot = path.resolve(process.cwd(), '..', 'assets', 'material_functions');
const materialFunctionType = 'a7ca55e7-0000-0001-0000-000000000011';
const materialFunctionImporter = 'a7ca55e7-0000-0002-0000-000000000012';

const functionFiles = () =>
  fs
    .readdirSync(builtInFunctionRoot, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith('.arcmatfn'))
    .map((entry) => entry.name)
    .sort();

const readFunction = (name: string) =>
  JSON.parse(fs.readFileSync(path.join(builtInFunctionRoot, name), 'utf8')) as MaterialFunctionAssetJson;

describe('built-in Material Functions', () => {
  it('ships the initial reusable color-source library', () => {
    expect(functionFiles()).toEqual([
      'channel_mask.arcmatfn',
      'checker.arcmatfn',
      'default_base_color.arcmatfn',
      'gradient.arcmatfn',
      'noise.arcmatfn',
      'texture_tint.arcmatfn',
      'tint_rgb.arcmatfn',
      'unpack_orm.arcmatfn',
      'uv_transform.arcmatfn',
    ]);
  });

  it.each(functionFiles())('%s is a graph-backed typed Material Function asset', (file) => {
    const asset = readFunction(file);
    expect(asset.kind).toBe('materialFunction');
    expect(asset.version).toBe(1);
    expect(isMaterialGraph(asset.graph)).toBe(true);
    const expectedOutputs: Record<string, Array<{ id: string; name: string; type: string }>> = {
      channel_mask: [{ id: 'value', name: 'Value', type: 'float' }],
      unpack_orm: [
        { id: 'ao', name: 'Ambient Occlusion', type: 'float' },
        { id: 'roughness', name: 'Roughness', type: 'float' },
        { id: 'metallic', name: 'Metallic', type: 'float' },
      ],
      uv_transform: [{ id: 'uv', name: 'UV', type: 'vec2' }],
    };
    const key = file.replace('.arcmatfn', '');
    expect(asset.outputs).toEqual(expectedOutputs[key] ?? [{ id: 'color', name: 'Color', type: 'vec3' }]);
    expect(asset.graph.nodes.filter((node) => node.type === 'functionOutput')).toHaveLength(1);
    expect(asset.graph.nodes.some((node) => node.type === 'functionSlot')).toBe(false);
  });

  it('lets the default base-color function own its tint and texture parameters', () => {
    const baseColor = readFunction('default_base_color.arcmatfn');
    expect(baseColor.inputs).toEqual([{ id: 'uv', name: 'UV', type: 'vec2' }]);
    expect(
      baseColor.graph.nodes.filter((node) => node.parameter?.exposed === true).map((node) => node.parameter?.name),
    ).toEqual(['Base Color Tint', 'Base Color Texture']);
    expect(baseColor.graph.nodes.some((node) => node.type === 'textureSample2D')).toBe(true);
    expect(baseColor.graph.nodes.some((node) => node.type === 'functionCall' && node.values.path === 'material_functions/tint_rgb.arcmatfn')).toBe(true);
  });

  it('gives every built-in a stable unique Material Function identity', () => {
    const guids = functionFiles().map((file) => {
      const metadata = JSON.parse(fs.readFileSync(path.join(builtInFunctionRoot, `${file}.arcmeta`), 'utf8')) as {
        guid: string;
        type: string;
        importer: string;
      };
      expect(metadata.type).toBe(materialFunctionType);
      expect(metadata.importer).toBe(materialFunctionImporter);
      expect(metadata.guid).toMatch(/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i);
      return metadata.guid;
    });
    expect(new Set(guids).size).toBe(guids.length);
  });

  it('authors Checker from world-space XZ coordinates', () => {
    const checker = readFunction('checker.arcmatfn');
    const worldPosition = checker.graph.nodes.find((node) => node.type === 'worldPosition');
    expect(worldPosition).toBeDefined();
    const sourcePins = checker.graph.connections
      .filter((connection) => connection.from.nodeId === worldPosition?.id)
      .map((connection) => connection.from.pin);
    expect(sourcePins).toEqual(expect.arrayContaining(['x', 'z']));
    expect(checker.inputs.map((input) => input.id)).toEqual(['colorA', 'colorB', 'cellSize', 'uv']);
  });

  it('authors Gradient and Noise from backend-neutral world-space graph operations', () => {
    for (const file of ['gradient.arcmatfn', 'noise.arcmatfn']) {
      const asset = readFunction(file);
      expect(asset.graph.nodes.some((node) => node.type === 'worldPosition')).toBe(true);
      expect(asset.graph.nodes.some((node) => node.type === 'dot')).toBe(true);
    }
    expect(readFunction('gradient.arcmatfn').inputs.map((input) => input.id)).toEqual([
      'colorA',
      'colorB',
      'direction',
      'scale',
      'offset',
      'uv',
    ]);
    expect(readFunction('noise.arcmatfn').inputs.map((input) => input.id)).toEqual([
      'scale',
      'seed',
      'contrast',
      'colorA',
      'colorB',
      'uv',
    ]);
  });
});
