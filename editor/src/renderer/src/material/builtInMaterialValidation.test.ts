import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { materialEditorParameters } from './materialCompiler';
import {
  isMaterialGraph,
  materialNodeDefinitions,
  materialScalarRange,
  type MaterialAssetJson,
  type MaterialGraph,
} from './materialGraphTypes';

const builtInMaterialRoot = path.resolve(process.cwd(), '..', 'assets', 'materials');

const builtInMaterialFiles = () =>
  fs
    .readdirSync(builtInMaterialRoot, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith('.arcmat'))
    .map((entry) => entry.name)
    .sort();

const readBuiltIn = (name: string) =>
  JSON.parse(fs.readFileSync(path.join(builtInMaterialRoot, name), 'utf8')) as MaterialAssetJson;

const contributingNodeIds = (graph: MaterialGraph) => {
  const incoming = new Map<string, string[]>();
  for (const connection of graph.connections) {
    const sources = incoming.get(connection.to.nodeId) ?? [];
    sources.push(connection.from.nodeId);
    incoming.set(connection.to.nodeId, sources);
  }

  const contributing = new Set<string>();
  const pending = graph.nodes.filter((node) => node.type === 'output').map((node) => node.id);
  while (pending.length > 0) {
    const nodeId = pending.pop()!;
    if (contributing.has(nodeId)) continue;
    contributing.add(nodeId);
    pending.push(...(incoming.get(nodeId) ?? []));
  }
  return contributing;
};

const directSemanticRangeMismatches = (graph: MaterialGraph) => {
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  const output = graph.nodes.find((node) => node.type === 'output');
  if (!output) return ['missing Material Output'];

  const outputPins = new Map(materialNodeDefinitions.output.inputs.map((pin) => [pin.id, pin]));
  return graph.connections.flatMap((connection) => {
    if (connection.to.nodeId !== output.id) return [];
    const expected = outputPins.get(connection.to.pin)?.semanticRange;
    if (!expected) return [];

    const source = byId.get(connection.from.nodeId);
    if (!source || source.type !== 'constant') return [];

    const range = materialScalarRange(source);
    if (range)
      return range.min < expected.min || range.max > expected.max
        ? [`${source.id} range ${range.min}..${range.max} -> ${connection.to.pin} ${expected.min}..${expected.max}`]
        : [];

    const value = source.values.value;
    return typeof value === 'number' && Number.isFinite(value) && (value < expected.min || value > expected.max)
      ? [`${source.id} value ${value} -> ${connection.to.pin} ${expected.min}..${expected.max}`]
      : [];
  });
};

describe('built-in material validation', () => {
  const files = builtInMaterialFiles();

  it('discovers the built-in material library', () => {
    expect(files.length).toBeGreaterThan(0);
  });

  it.each(files)('%s stays graph-valid and authoring-clean', (file) => {
    const asset = readBuiltIn(file);
    expect(asset.version, `${file}: authored schema`).toBe(4);
    expect(isMaterialGraph(asset.graph), `${file}: valid material graph`).toBe(true);

    const graph = asset.graph!;
    const contributing = contributingNodeIds(graph);
    const exposed = graph.nodes.filter((node) => node.parameter?.exposed === true);
    const presented = new Set(materialEditorParameters(graph).map((parameter) => parameter.nodeId));

    for (const node of exposed) {
      expect(contributing.has(node.id), `${file}: exposed parameter "${node.parameter?.name}" must affect output`).toBe(
        true,
      );
      if (node.type === 'functionCall' || node.type === 'functionSlot') {
        expect(typeof node.values.slotId, `${file}: exposed function must have a slot ID`).toBe('string');
        expect(typeof node.values.path, `${file}: exposed function must reference a function`).toBe('string');
      } else {
        expect(presented.has(node.id), `${file}: exposed parameter "${node.parameter?.name}" must be authorable`).toBe(
          true,
        );
      }
    }

    for (const node of graph.nodes) {
      if (node.type !== 'constant') continue;
      const hasMin = Object.hasOwn(node.values, 'min');
      const hasMax = Object.hasOwn(node.values, 'max');
      expect(hasMin, `${file}: Scalar "${node.id}" must author both range bounds or neither`).toBe(hasMax);
      if (!hasMin) continue;

      const range = materialScalarRange(node);
      expect(range, `${file}: Scalar "${node.id}" has an invalid range`).not.toBeNull();
      const value = node.values.value;
      expect(typeof value, `${file}: Scalar "${node.id}" must have a numeric value`).toBe('number');
      if (range && typeof value === 'number') {
        expect(Number.isFinite(value), `${file}: Scalar "${node.id}" value must be finite`).toBe(true);
        expect(value, `${file}: Scalar "${node.id}" is below its authored range`).toBeGreaterThanOrEqual(range.min);
        expect(value, `${file}: Scalar "${node.id}" is above its authored range`).toBeLessThanOrEqual(range.max);
      }
    }

    expect(directSemanticRangeMismatches(graph), `${file}: direct Material Output scalar domains`).toEqual([]);
  });
});
