import type { AssetItem } from '../services/editorHostTypes';
import {
  materialEditorParameters,
  type MaterialEditorParameterKind,
} from './materialCompiler';
import {
  materialGraphFromAsset,
  type MaterialAssetJson,
  type MaterialFunctionAssetJson,
  type MaterialFunctionPin,
  type MaterialGraphNode,
  type MaterialGraphValueType,
} from './materialGraphTypes';
import {
  materialFunctionSlotParameterId,
  materialParameterId,
  type MaterialInstanceAssetReference,
} from './materialInstancePersistence';

export type MaterialInstanceParentParameter = {
  id: string;
  nodeId: string;
  name: string;
  type: MaterialGraphValueType;
  editorKind: MaterialEditorParameterKind;
  range?: { min: number; max: number };
  value: unknown;
};

export type MaterialInstanceFunctionOption = {
  asset: AssetItem;
  reference: MaterialInstanceAssetReference;
  document: MaterialFunctionAssetJson;
};

export type MaterialInstanceFunctionSlot = {
  id: string;
  name: string;
  defaultFunction: MaterialInstanceFunctionOption | null;
  inputs: MaterialFunctionPin[];
  outputs: MaterialFunctionPin[];
  compatibleFunctions: MaterialInstanceFunctionOption[];
};

export type MaterialInstanceParentModel = {
  asset: AssetItem;
  parameters: MaterialInstanceParentParameter[];
  functionSlots: MaterialInstanceFunctionSlot[];
};

const normalizePath = (value: string) =>
  value.trim().replaceAll('\\', '/').replace(/\/+/g, '/').replace(/^\.\//, '').replace(/^\//, '').toLowerCase();

const pathMatches = (candidate: string | undefined, hint: string) => {
  const left = normalizePath(candidate ?? '');
  const right = normalizePath(hint);
  return Boolean(left && right && (left === right || left.endsWith(`/${right}`) || right.endsWith(`/${left}`)));
};

export const materialAssetReference = (asset: AssetItem): MaterialInstanceAssetReference | null => {
  const guid = asset.guid?.trim();
  const pathHint = (asset.sourcePath || asset.path).trim().replaceAll('\\', '/');
  return guid && pathHint ? { guid, pathHint } : null;
};

export const findMaterialInstanceAsset = (
  assets: readonly AssetItem[],
  reference: MaterialInstanceAssetReference,
  kind?: AssetItem['kind'],
) =>
  assets.find(
    (asset) =>
      (!kind || asset.kind === kind) &&
      ((asset.guid && asset.guid === reference.guid) ||
        pathMatches(asset.sourcePath, reference.pathHint) ||
        pathMatches(asset.path, reference.pathHint)),
  ) ?? null;

const parameterValue = (node: MaterialGraphNode): unknown => {
  if (node.type === 'textureSample' || node.type === 'textureSample2D')
    return typeof node.values.texture === 'string' ? node.values.texture : '';
  if (typeof node.values.value === 'number') return node.values.value;
  if (Array.isArray(node.values.value)) return [...node.values.value];
  return 0;
};

const readAsset = async <T>(asset: AssetItem): Promise<T> => {
  const path = asset.sourcePath || asset.path;
  const scope = asset.scope === 'builtin' ? 'builtin' : 'project';
  const file = await window.arc.projects.readText(path, scope);
  return JSON.parse(file.text) as T;
};

const pinMap = (pins: readonly MaterialFunctionPin[]) => new Map(pins.map((pin) => [pin.id, pin]));

export const materialFunctionCompatibleWithSlot = (
  baseInputs: readonly MaterialFunctionPin[],
  baseOutputs: readonly MaterialFunctionPin[],
  replacement: MaterialFunctionAssetJson,
): boolean => {
  const replacementInputs = pinMap(replacement.inputs);
  const replacementOutputs = pinMap(replacement.outputs);

  for (const input of baseInputs) {
    if (input.default !== undefined) continue;
    const candidate = replacementInputs.get(input.id);
    if (!candidate || candidate.type !== input.type) return false;
  }
  for (const output of baseOutputs) {
    const candidate = replacementOutputs.get(output.id);
    if (!candidate || candidate.type !== output.type) return false;
  }
  const baseInputMap = pinMap(baseInputs);
  for (const input of replacement.inputs) {
    if (baseInputMap.has(input.id)) continue;
    if (input.default === undefined) return false;
  }
  return true;
};

export const loadMaterialFunctionOptions = async (assets: readonly AssetItem[]) => {
  const options = await Promise.all(
    assets
      .filter((asset) => asset.kind === 'materialFunction' && asset.scope !== 'procedural')
      .map(async (asset): Promise<MaterialInstanceFunctionOption | null> => {
        const reference = materialAssetReference(asset);
        if (!reference) return null;
        try {
          const document = await readAsset<MaterialFunctionAssetJson>(asset);
          if (document.kind !== 'materialFunction' || document.version !== 1) return null;
          return { asset, reference, document };
        } catch {
          return null;
        }
      }),
  );
  return options.filter((option): option is MaterialInstanceFunctionOption => option !== null);
};

export const loadMaterialInstanceParentModel = async (
  parent: MaterialInstanceAssetReference,
  assets: readonly AssetItem[],
): Promise<MaterialInstanceParentModel> => {
  const asset = findMaterialInstanceAsset(assets, parent, 'material');
  if (!asset) throw new Error('Parent Material is unavailable.');
  const document = await readAsset<MaterialAssetJson>(asset);
  if (typeof document.shaderPath === 'string' && document.shaderPath.trim()) {
    return { asset, parameters: [], functionSlots: [] };
  }

  const graph = materialGraphFromAsset(document);
  const parameters = materialEditorParameters(graph).flatMap((parameter) => {
    const node = graph.nodes.find((candidate) => candidate.id === parameter.nodeId);
    if (!node) return [];
    return [
      {
        id: materialParameterId(node.id),
        nodeId: node.id,
        name: parameter.name,
        type: parameter.type,
        editorKind: parameter.editorKind,
        range: parameter.range,
        value: parameterValue(node),
      },
    ];
  });

  const functionOptions = await loadMaterialFunctionOptions(assets);
  const functionSlots: MaterialInstanceFunctionSlot[] = graph.nodes.flatMap((node) => {
    if (node.type !== 'functionSlot') return [];
    const id = typeof node.values.slotId === 'string' ? node.values.slotId.trim() : '';
    const name = typeof node.values.name === 'string' && node.values.name.trim() ? node.values.name.trim() : 'Function Slot';
    const defaultPath = typeof node.values.path === 'string' ? node.values.path : '';
    const inputs = Array.isArray(node.values.inputPins) ? (node.values.inputPins as MaterialFunctionPin[]) : [];
    const outputs = Array.isArray(node.values.outputPins) ? (node.values.outputPins as MaterialFunctionPin[]) : [];
    if (!id || !defaultPath) return [];
    const defaultFunction =
      functionOptions.find((option) => pathMatches(option.reference.pathHint, defaultPath)) ?? null;
    const compatibleFunctions = functionOptions.filter((option) =>
      materialFunctionCompatibleWithSlot(inputs, outputs, option.document),
    );
    return [{ id, name, defaultFunction, inputs, outputs, compatibleFunctions }];
  });

  return { asset, parameters, functionSlots };
};

export const selectedFunctionExtraParameters = (
  slot: MaterialInstanceFunctionSlot,
  selected: MaterialInstanceFunctionOption,
) => {
  const baseInputs = new Set(slot.inputs.map((input) => input.id));
  return selected.document.inputs.flatMap((input) => {
    if (baseInputs.has(input.id) || input.default === undefined) return [];
    return [
      {
        id: materialFunctionSlotParameterId(slot.id, selected.reference.guid, input.id),
        pinId: input.id,
        name: input.name,
        type: input.type,
        value: input.default,
      },
    ];
  });
};
