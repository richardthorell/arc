import { useEffect, useMemo, useState } from 'react';
import { RotateCcw, TriangleAlert } from 'lucide-react';

import { materialEditorParameters, type MaterialEditorParameterKind } from '../material/materialCompiler';
import { materialFunctionCompatibleWithSlot } from '../material/materialInstanceAuthoring';
import {
  deserializeMaterialInstanceAsset,
  materialFunctionSlotParameterId,
  materialParameterId,
} from '../material/materialInstancePersistence';
import {
  materialGraphFromAsset,
  type MaterialAssetJson,
  type MaterialFunctionAssetJson,
  type MaterialFunctionPin,
  type MaterialGraphNode,
  type MaterialGraphValueType,
} from '../material/materialGraphTypes';
import { UiColorControl, UiNumericInput, UiSelect, UiSlider } from '../ui';
import { hostAssetReference, type HostAssetReference } from '../services/assetReferences';
import { TexturePicker, type AssetThumbnailProvider } from './AssetPicker';
import type { HostEntityId, HostResponse, Vec4 } from './inspectorTypes';
import { NumberControl } from './InspectorControls';

import './inspectorPolish.css';

type MaterialParameterAsset = {
  id: string;
  guid?: string;
  typeId?: string;
  name: string;
  path: string;
  sourcePath?: string;
  kind: string;
  status: 'unknown' | 'queued' | 'ready' | 'dirty' | 'source' | 'stale' | 'importing' | 'failed' | 'missing';
  scope?: 'builtin' | 'project' | 'user' | 'organization' | 'procedural';
  readOnly?: boolean;
};

type InstanceOverride = {
  parameterId?: string;
  slotId?: string;
  name: string;
  type: MaterialGraphValueType;
  kind: MaterialEditorParameterKind;
  value?: number[];
  texture?: string;
  textureReference?: HostAssetReference;
};

type DisplayParameter = {
  nodeId: string;
  slotId?: string;
  name: string;
  type: MaterialGraphValueType;
  editorKind: MaterialEditorParameterKind;
  range?: { min: number; max: number };
  values: number[];
  texture: string;
};

type DisplayFunctionOption = {
  guid: string;
  typeId?: string;
  path: string;
  name: string;
  document: MaterialFunctionAssetJson;
};

type DisplayFunctionSlot = {
  id: string;
  name: string;
  inputs: MaterialFunctionPin[];
  outputs: MaterialFunctionPin[];
  defaultGuid: string;
  resetGuid: string;
  selectedGuid: string;
  options: DisplayFunctionOption[];
  parameters: DisplayParameter[];
};

type RuntimeFunctionOverride = {
  kind: 'function';
  slotId: string;
  function: { guid: string; expectedType?: string; pathHint: string };
};

type ParameterState =
  | { status: 'idle'; parameters: DisplayParameter[]; functionSlots: DisplayFunctionSlot[] }
  | { status: 'loading'; parameters: DisplayParameter[]; functionSlots: DisplayFunctionSlot[] }
  | { status: 'ready'; parameters: DisplayParameter[]; functionSlots: DisplayFunctionSlot[] }
  | { status: 'custom'; parameters: DisplayParameter[]; functionSlots: DisplayFunctionSlot[] }
  | { status: 'error'; parameters: DisplayParameter[]; functionSlots: DisplayFunctionSlot[] };

type SelectedMaterialSnapshot = {
  entity: HostEntityId;
  selectionCount?: number;
  meshRenderer?: { materialName?: string };
};

const emptyState: ParameterState = { status: 'idle', parameters: [], functionSlots: [] };
const componentLabels = ['X', 'Y', 'Z', 'W'];
const instanceMarker = '__arc_instance_overrides__';
const parameterCommandPrefix = '__arc_primitive_parameter__/__arc_material_parameter__';
const functionCommandPrefix = '__arc_primitive_parameter__/__arc_material_function__';

const normalizePath = (value: string) =>
  value
    .trim()
    .replaceAll('\\', '/')
    .replace(/\/+/g, '/')
    .replace(/^\.\//, '')
    .replace(/^\/|\/$/g, '');

const builtinReferencePath = (value: string) => normalizePath(value).replace(/^builtin\//i, '');

const projectRelativeMaterialPath = async (materialPath: string, scope: 'builtin' | 'project') => {
  const normalized = normalizePath(materialPath);
  if (scope !== 'project' || !normalized || /^[a-z]:\//i.test(normalized)) return normalized;

  const projects = window.arc?.projects;
  if (!projects || typeof projects.snapshot !== 'function') return normalized;

  try {
    const snapshot = await projects.snapshot();
    const project = snapshot?.activeProject;
    if (!project) return normalized;

    const roots = [project.descriptor.paths.content, ...(project.descriptor.assetRoots ?? [])]
      .map(normalizePath)
      .filter(Boolean);
    if (roots.some((root) => normalized.toLocaleLowerCase().startsWith(`${root.toLocaleLowerCase()}/`))) {
      return normalized;
    }

    const contentRoot = roots[0] || 'Content';
    const resolved = normalizePath(`${contentRoot}/${normalized}`);
    console.info('[material-flow] inspector resolved material path', {
      registryPath: materialPath,
      projectPath: resolved,
    });
    return resolved;
  } catch (error) {
    console.warn('[material-flow] inspector could not resolve project material path', error);
    return normalized;
  }
};

const parameterValues = (node: MaterialGraphNode): number[] => {
  if (typeof node.values.value === 'number' && Number.isFinite(node.values.value)) return [node.values.value];
  if (!Array.isArray(node.values.value)) return [];
  return node.values.value.map((value) => (typeof value === 'number' && Number.isFinite(value) ? value : 0));
};

const parameterTexture = (node: MaterialGraphNode) =>
  typeof node.values.texture === 'string' ? node.values.texture : '';

const bytesToHex = (text: string) =>
  Array.from(new TextEncoder().encode(text), (byte) => byte.toString(16).padStart(2, '0')).join('');

const hexToText = (hex: string) => {
  if (!hex || hex.length % 2 !== 0 || !/^[0-9a-f]+$/i.test(hex)) return '';
  const bytes = new Uint8Array(hex.length / 2);
  for (let index = 0; index < bytes.length; ++index)
    bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  return new TextDecoder().decode(bytes);
};

const persistedRuntimeStateFromMaterialName = (name: string | undefined): unknown[] => {
  const marker = name?.indexOf(instanceMarker) ?? -1;
  if (!name || marker < 0) return [];
  try {
    const parsed = JSON.parse(hexToText(name.slice(marker + instanceMarker.length)) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

const overridesFromMaterialName = (name: string | undefined): InstanceOverride[] =>
  persistedRuntimeStateFromMaterialName(name).filter((entry): entry is InstanceOverride =>
    Boolean(entry && typeof entry === 'object' && typeof (entry as InstanceOverride).name === 'string'),
  );

const functionOverridesFromMaterialName = (name: string | undefined): RuntimeFunctionOverride[] =>
  persistedRuntimeStateFromMaterialName(name).filter((entry): entry is RuntimeFunctionOverride =>
    Boolean(
      entry &&
      typeof entry === 'object' &&
      (entry as RuntimeFunctionOverride).kind === 'function' &&
      typeof (entry as RuntimeFunctionOverride).slotId === 'string',
    ),
  );

const pathMatches = (candidate: string, hint: string) => {
  const left = normalizePath(candidate).toLocaleLowerCase();
  const right = normalizePath(hint).toLocaleLowerCase();
  return Boolean(left && right && (left === right || left.endsWith(`/${right}`) || right.endsWith(`/${left}`)));
};

const functionEditorKind = (pin: MaterialFunctionPin): MaterialEditorParameterKind =>
  /^colou?r\b/i.test(pin.name) && (pin.type === 'vec3' || pin.type === 'vec4')
    ? 'color'
    : pin.type === 'float'
      ? 'scalar'
      : 'vector';

const functionParameterValues = (value: unknown): number[] =>
  typeof value === 'number'
    ? [value]
    : Array.isArray(value)
      ? value.map((component) => (typeof component === 'number' && Number.isFinite(component) ? component : 0))
      : [];

const numericField = (label: string, range?: { min: number; max: number }) => ({
  label,
  precision: 3,
  step: 0.01,
  scrubSensitivity: 0.005,
  min: range?.min,
  max: range?.max,
});

const clampParameterValue = (value: number, range?: { min: number; max: number }) =>
  range ? Math.min(range.max, Math.max(range.min, value)) : value;

export function MaterialParameterSubsection({
  assets,
  mixed = false,
  referenceMode = 'path',
  thumbnailProvider,
  value,
}: {
  assets: ReadonlyArray<MaterialParameterAsset>;
  mixed?: boolean;
  referenceMode?: 'path' | 'guid';
  thumbnailProvider?: AssetThumbnailProvider;
  value: string;
}) {
  const selected = useMemo(
    () =>
      assets.find(
        (asset) =>
          (asset.kind === 'material' || asset.kind === 'materialInstance') &&
          (referenceMode === 'guid' ? (asset.guid || asset.id) === value : asset.path === value),
      ),
    [assets, referenceMode, value],
  );
  const materialPath = selected?.path ?? (referenceMode === 'path' && /\.arcmat(?:inst)?$/i.test(value) ? value : '');
  const materialScope = selected?.scope === 'builtin' ? 'builtin' : 'project';
  const procedural = selected?.scope === 'procedural';
  const [state, setState] = useState<ParameterState>(emptyState);
  const [overrides, setOverrides] = useState<InstanceOverride[]>([]);
  const [mutationError, setMutationError] = useState('');

  useEffect(() => {
    let active = true;
    if (!value || mixed || !materialPath || procedural) {
      setState(emptyState);
      setOverrides([]);
      return () => {
        active = false;
      };
    }

    setState({ status: 'loading', parameters: [], functionSlots: [] });
    void (async () => {
      try {
        const resolvedMaterialPath = await projectRelativeMaterialPath(materialPath, materialScope);
        const [file, selection] = await Promise.all([
          window.arc.projects.readText(resolvedMaterialPath, materialScope),
          window.arc.host?.query('entity.selected') as Promise<HostResponse<SelectedMaterialSnapshot>> | undefined,
        ]);
        if (!active) return;
        let materialAsset: MaterialAssetJson;
        let instanceDefaults = new Map<string, unknown>();
        const instanceFunctionOverrides = new Map<
          string,
          { guid: string; pathHint: string; inputOverrides: Array<{ pinId: string; value: unknown }> }
        >();
        if (selected?.kind === 'materialInstance' || /\.arcmatinst$/i.test(materialPath)) {
          const instance = deserializeMaterialInstanceAsset(file.text);
          if (!instance) throw new Error('Material Instance metadata is unavailable');
          const parent = assets.find(
            (candidate) =>
              candidate.kind === 'material' &&
              ((candidate.guid && candidate.guid === instance.parent.guid) ||
                candidate.path
                  .replaceAll('\\', '/')
                  .toLocaleLowerCase()
                  .endsWith(instance.parent.pathHint.replaceAll('\\', '/').toLocaleLowerCase())),
          );
          if (!parent) throw new Error('Material Instance parent is unavailable');
          const parentPath = await projectRelativeMaterialPath(
            parent.path,
            parent.scope === 'builtin' ? 'builtin' : 'project',
          );
          const parentFile = await window.arc.projects.readText(
            parentPath,
            parent.scope === 'builtin' ? 'builtin' : 'project',
          );
          materialAsset = JSON.parse(parentFile.text) as MaterialAssetJson;
          instanceDefaults = new Map(instance.parameterOverrides.map((entry) => [entry.parameterId, entry.value]));
          for (const entry of instance.functionOverrides)
            instanceFunctionOverrides.set(entry.slotId, {
              guid: entry.function.guid,
              pathHint: entry.function.pathHint,
              inputOverrides: entry.inputOverrides,
            });
        } else {
          materialAsset = JSON.parse(file.text) as MaterialAssetJson;
        }

        const customShader = typeof materialAsset.shaderPath === 'string' ? materialAsset.shaderPath.trim() : '';
        if (customShader) {
          setState({ status: 'custom', parameters: [], functionSlots: [] });
          return;
        }

        const graph = materialGraphFromAsset(materialAsset);
        const parameters = materialEditorParameters(graph).map((parameter) => {
          const node = graph.nodes.find((candidate) => candidate.id === parameter.nodeId);
          const instanceValue = node ? instanceDefaults.get(materialParameterId(node.id)) : undefined;
          const authoredValues =
            instanceValue === undefined
              ? node
                ? parameterValues(node)
                : []
              : typeof instanceValue === 'number'
                ? [instanceValue]
                : Array.isArray(instanceValue)
                  ? instanceValue.map((value) => (typeof value === 'number' ? value : 0))
                  : [];
          const authoredTexture =
            instanceValue === undefined
              ? node
                ? parameterTexture(node)
                : ''
              : typeof instanceValue === 'string'
                ? instanceValue
                : '';
          return {
            ...parameter,
            nodeId: materialParameterId(parameter.nodeId),
            values: authoredValues,
            texture: authoredTexture,
          };
        });
        const runtimeMaterialName = selection?.succeeded ? selection.payload?.meshRenderer?.materialName : undefined;
        const runtimeFunctions = functionOverridesFromMaterialName(runtimeMaterialName);
        setOverrides(selection?.succeeded ? overridesFromMaterialName(runtimeMaterialName) : []);

        const functionOptions = (
          await Promise.all(
            assets
              .filter((asset) => asset.kind === 'materialFunction' && asset.guid)
              .map(async (asset): Promise<DisplayFunctionOption | null> => {
                try {
                  const scope = asset.scope === 'builtin' ? 'builtin' : 'project';
                  const authoringPath = scope === 'builtin' ? asset.path : asset.sourcePath || asset.path;
                  const referencePath =
                    scope === 'builtin' ? builtinReferencePath(authoringPath) : asset.sourcePath || asset.path;
                  const path = await projectRelativeMaterialPath(authoringPath, scope);
                  const source = await window.arc.projects.readText(path, scope);
                  const document = JSON.parse(source.text) as MaterialFunctionAssetJson;
                  if (document.kind !== 'materialFunction' || document.version !== 1 || !asset.guid) return null;
                  return { guid: asset.guid, typeId: asset.typeId, path: referencePath, name: document.name, document };
                } catch {
                  return null;
                }
              }),
          )
        ).filter((option): option is DisplayFunctionOption => option !== null);

        const functionSlots = graph.nodes.flatMap((node): DisplayFunctionSlot[] => {
          const authoredReferences =
            node.type === 'functionCall' && Array.isArray(node.values.functions)
              ? node.values.functions.flatMap((reference) => {
                  if (!reference || typeof reference !== 'object') return [];
                  const path = (reference as { path?: unknown }).path;
                  return typeof path === 'string' && path.trim() ? [path] : [];
                })
              : [];
          // A single referenced function is a fixed graph dependency, not a choice.
          const selectableCall = node.type === 'functionCall' && authoredReferences.length > 1;
          if (node.type !== 'functionSlot' && !selectableCall) return [];
          const id = typeof node.values.slotId === 'string' ? node.values.slotId.trim() : '';
          const name =
            typeof node.values.name === 'string' && node.values.name.trim() ? node.values.name.trim() : 'Function';
          const defaultPath = typeof node.values.path === 'string' ? node.values.path : '';
          const inputs = Array.isArray(node.values.inputPins) ? (node.values.inputPins as MaterialFunctionPin[]) : [];
          const outputs = Array.isArray(node.values.outputPins)
            ? (node.values.outputPins as MaterialFunctionPin[])
            : [];
          if (!id || !defaultPath) return [];
          const compatible = functionOptions.filter(
            (option) =>
              materialFunctionCompatibleWithSlot(inputs, outputs, option.document) &&
              (!selectableCall || authoredReferences.some((reference) => pathMatches(option.path, reference))),
          );
          const defaultOption = compatible.find((option) => pathMatches(option.path, defaultPath));
          if (!defaultOption) return [];

          const authored = instanceFunctionOverrides.get(id);
          const runtime = runtimeFunctions.find((entry) => entry.slotId === id);
          const selectedGuid = runtime?.function.guid || authored?.guid || defaultOption.guid;
          const selectedOption = compatible.find((option) => option.guid === selectedGuid) ?? defaultOption;
          const authoredInputs = new Map((authored?.inputOverrides ?? []).map((entry) => [entry.pinId, entry.value]));
          const baseInputs = new Set(inputs.map((input) => input.id));
          const inputParameters = selectedOption.document.inputs.flatMap((pin): DisplayParameter[] => {
            if (baseInputs.has(pin.id) || pin.default === undefined) return [];
            const parameterId = materialFunctionSlotParameterId(id, selectedOption.guid, pin.id);
            const runtimeOverride = overridesFromMaterialName(runtimeMaterialName).find(
              (entry) => entry.parameterId === parameterId,
            );
            const value = runtimeOverride?.value ?? authoredInputs.get(pin.id) ?? pin.default;
            return [
              {
                nodeId: parameterId,
                slotId: id,
                name: pin.name,
                type: pin.type,
                editorKind: functionEditorKind(pin),
                values: functionParameterValues(value),
                texture: '',
              },
            ];
          });
          const graphParameters = materialEditorParameters(selectedOption.document.graph).map((parameter) => {
            const functionNode = selectedOption.document.graph.nodes.find(
              (candidate) => candidate.id === parameter.nodeId,
            );
            const parameterId = materialParameterId(`${id}::${parameter.nodeId}`);
            return {
              ...parameter,
              nodeId: parameterId,
              slotId: id,
              values: functionNode ? parameterValues(functionNode) : [],
              texture: functionNode ? parameterTexture(functionNode) : '',
            };
          });
          const extraParameters = [...graphParameters, ...inputParameters];

          return [
            {
              id,
              name,
              inputs,
              outputs,
              defaultGuid: defaultOption.guid,
              resetGuid: authored?.guid || defaultOption.guid,
              selectedGuid: selectedOption.guid,
              options: compatible,
              parameters: extraParameters,
            },
          ];
        });

        setState({ status: 'ready', parameters, functionSlots });
      } catch {
        if (active) setState({ status: 'error', parameters: [], functionSlots: [] });
      }
    })();

    return () => {
      active = false;
    };
  }, [assets, materialPath, materialScope, mixed, procedural, selected?.kind, value]);

  const overrideFor = (parameter: DisplayParameter) =>
    overrides.find(
      (entry) => entry.parameterId === parameter.nodeId || (!entry.parameterId && entry.name === parameter.name),
    );
  const effectiveValues = (parameter: DisplayParameter) => overrideFor(parameter)?.value ?? parameter.values;
  const effectiveTexture = (parameter: DisplayParameter) => overrideFor(parameter)?.texture ?? parameter.texture;

  const updateLocalOverride = (parameter: DisplayParameter, next: InstanceOverride | null) => {
    setOverrides((current) => {
      const filtered = current.filter(
        (entry) => entry.parameterId !== parameter.nodeId && (entry.parameterId || entry.name !== parameter.name),
      );
      return next ? [...filtered, next] : filtered;
    });
  };

  const commitOverride = async (parameter: DisplayParameter, next: InstanceOverride | null) => {
    updateLocalOverride(parameter, next);
    setMutationError('');
    if (!window.arc?.host) return;
    try {
      const selectedResponse = (await window.arc.host.query(
        'entity.selected',
      )) as HostResponse<SelectedMaterialSnapshot>;
      if (!selectedResponse.succeeded || !selectedResponse.payload)
        throw new Error(selectedResponse.error || 'Selected entity is unavailable');
      const payload = next
        ? { ...next, parameterId: parameter.nodeId, ...(parameter.slotId ? { slotId: parameter.slotId } : {}) }
        : {
            parameterId: parameter.nodeId,
            ...(parameter.slotId ? { slotId: parameter.slotId } : {}),
            name: parameter.name,
            type: parameter.type,
            kind: parameter.editorKind,
            reset: true,
          };
      const path = `${parameterCommandPrefix}${bytesToHex(JSON.stringify(payload))}/0`;
      const response = (await window.arc.host.command('entity.setMaterial', {
        entity: selectedResponse.payload.entity,
        applyToSelection: (selectedResponse.payload.selectionCount ?? 1) > 1,
        path,
      })) as HostResponse;
      if (!response.succeeded) throw new Error(response.error || 'Material parameter override failed');
    } catch (error) {
      setMutationError(error instanceof Error ? error.message : String(error));
    }
  };

  const commitFunction = async (slot: DisplayFunctionSlot, option: DisplayFunctionOption) => {
    setMutationError('');
    setState((current) =>
      current.status !== 'ready'
        ? current
        : {
            ...current,
            functionSlots: current.functionSlots.map((candidate) =>
              candidate.id !== slot.id
                ? candidate
                : {
                    ...candidate,
                    selectedGuid: option.guid,
                    parameters: [
                      ...materialEditorParameters(option.document.graph).map((parameter) => {
                        const functionNode = option.document.graph.nodes.find((node) => node.id === parameter.nodeId);
                        return {
                          ...parameter,
                          nodeId: materialParameterId(`${candidate.id}::${parameter.nodeId}`),
                          slotId: candidate.id,
                          values: functionNode ? parameterValues(functionNode) : [],
                          texture: functionNode ? parameterTexture(functionNode) : '',
                        };
                      }),
                      ...option.document.inputs.flatMap((pin): DisplayParameter[] => {
                        if (candidate.inputs.some((input) => input.id === pin.id) || pin.default === undefined)
                          return [];
                        return [
                          {
                            nodeId: materialFunctionSlotParameterId(candidate.id, option.guid, pin.id),
                            slotId: candidate.id,
                            name: pin.name,
                            type: pin.type,
                            editorKind: functionEditorKind(pin),
                            values: functionParameterValues(pin.default),
                            texture: '',
                          },
                        ];
                      }),
                    ],
                  },
            ),
          },
    );
    setOverrides((current) =>
      current.filter((entry) => !slot.parameters.some((parameter) => parameter.nodeId === entry.parameterId)),
    );
    if (!window.arc?.host) return;
    try {
      const selectedResponse = (await window.arc.host.query(
        'entity.selected',
      )) as HostResponse<SelectedMaterialSnapshot>;
      if (!selectedResponse.succeeded || !selectedResponse.payload)
        throw new Error(selectedResponse.error || 'Selected entity is unavailable');
      const payload = {
        slotId: slot.id,
        function: {
          guid: option.guid,
          ...(option.typeId ? { expectedType: option.typeId } : {}),
          pathHint: option.path,
        },
        reset: option.guid === slot.resetGuid,
      };
      const path = `${functionCommandPrefix}${bytesToHex(JSON.stringify(payload))}/0`;
      const response = (await window.arc.host.command('entity.setMaterial', {
        entity: selectedResponse.payload.entity,
        applyToSelection: (selectedResponse.payload.selectionCount ?? 1) > 1,
        path,
      })) as HostResponse;
      if (!response.succeeded) throw new Error(response.error || 'Material function override failed');
    } catch (error) {
      setMutationError(error instanceof Error ? error.message : String(error));
    }
  };

  if (!value || mixed || !materialPath || procedural) return null;

  const summary =
    state.status === 'loading'
      ? 'Loading…'
      : state.status === 'ready'
        ? `${state.parameters.length} exposed${overrides.length ? ` · ${overrides.length} overridden` : ''}`
        : state.status === 'custom'
          ? 'Cook-reflected'
          : '';

  return (
    <section className="inspector-subsection inspector-material-parameters" aria-label="Material parameters">
      <header className="inspector-subsection-title">
        <span>Material Parameters</span>
        {summary && <small>{summary}</small>}
      </header>
      {state.status === 'ready' && state.functionSlots.length > 0 && (
        <div className="inspector-material-function-list">
          {state.functionSlots.map((slot) => {
            const selectedOption =
              slot.options.find((option) => option.guid === slot.selectedGuid) ??
              slot.options.find((option) => option.guid === slot.defaultGuid);
            return (
              <div className="inspector-material-function" key={slot.id}>
                <div className="inspector-material-function-selector">
                  <span className="inspector-property-label">{slot.name}</span>
                  <UiSelect
                    ariaLabel={`${slot.name} function`}
                    options={slot.options.map((option) => ({ value: option.guid, label: option.name }))}
                    value={selectedOption?.guid ?? slot.defaultGuid}
                    onValueChange={(guid) => {
                      const option = slot.options.find((candidate) => candidate.guid === guid);
                      if (option) void commitFunction(slot, option);
                    }}
                  />
                </div>
                {slot.parameters.length > 0 && (
                  <div className="inspector-material-function-parameters">
                    {slot.parameters.map((parameter) => {
                      const override = overrideFor(parameter);
                      const values = effectiveValues(parameter);
                      const reset = override ? (
                        <button
                          aria-label={`Reset ${slot.name} ${parameter.name}`}
                          className="inspector-field-reset"
                          onClick={() => void commitOverride(parameter, null)}
                          title="Revert to function default"
                          type="button"
                        >
                          <RotateCcw aria-hidden="true" size={12} />
                        </button>
                      ) : null;

                      if (parameter.editorKind === 'texture') {
                        const textureValue = effectiveTexture(parameter);
                        return (
                          <div className="inspector-material-parameter" key={parameter.nodeId}>
                            <TexturePicker
                              allowEmpty
                              assets={assets}
                              label={parameter.name}
                              thumbnailProvider={thumbnailProvider}
                              value={textureValue}
                              onChange={(texture, asset) =>
                                void commitOverride(parameter, {
                                  parameterId: parameter.nodeId,
                                  name: parameter.name,
                                  type: parameter.type,
                                  kind: parameter.editorKind,
                                  texture,
                                  ...(asset ? { textureReference: hostAssetReference(asset) ?? undefined } : {}),
                                })
                              }
                            />
                            {reset}
                          </div>
                        );
                      }

                      if (parameter.editorKind === 'color') {
                        const rgba: Vec4 = {
                          x: values[0] ?? 0,
                          y: values[1] ?? 0,
                          z: values[2] ?? 0,
                          w: parameter.type === 'vec4' ? (values[3] ?? 1) : 1,
                        };
                        const colorOverride = (next: Vec4): InstanceOverride => ({
                          parameterId: parameter.nodeId,
                          name: parameter.name,
                          type: parameter.type,
                          kind: parameter.editorKind,
                          value:
                            parameter.type === 'vec4' ? [next.x, next.y, next.z, next.w] : [next.x, next.y, next.z],
                        });
                        return (
                          <div className="inspector-material-parameter" key={parameter.nodeId}>
                            <span className="inspector-property-label">{parameter.name}</span>
                            <UiColorControl
                              allowAlpha={parameter.type === 'vec4'}
                              label={parameter.name}
                              value={rgba}
                              onPreview={(next) => updateLocalOverride(parameter, colorOverride(next))}
                              onCommit={(next) => void commitOverride(parameter, colorOverride(next))}
                            />
                            {reset}
                          </div>
                        );
                      }

                      if (parameter.editorKind === 'scalar') {
                        const nextOverride = (next: number): InstanceOverride => ({
                          parameterId: parameter.nodeId,
                          name: parameter.name,
                          type: parameter.type,
                          kind: parameter.editorKind,
                          value: [next],
                        });
                        return (
                          <div className="inspector-material-parameter" key={parameter.nodeId}>
                            <div className="inspector-material-scalar-control">
                              <NumberControl
                                field={numericField(parameter.name)}
                                value={values[0] ?? 0}
                                onPreview={(next) => updateLocalOverride(parameter, nextOverride(next))}
                                onCommit={(next) => void commitOverride(parameter, nextOverride(next))}
                              />
                            </div>
                            {reset}
                          </div>
                        );
                      }

                      return (
                        <div className="inspector-material-parameter" key={parameter.nodeId}>
                          <span className="inspector-property-label">{parameter.name}</span>
                          <div className="inspector-material-parameter-values">
                            {values.map((parameterValue, index) => (
                              <UiNumericInput
                                ariaLabel={`${slot.name} ${parameter.name} ${componentLabels[index]}`}
                                key={index}
                                precision={3}
                                scrubClassName={`axis-${componentLabels[index].toLocaleLowerCase()}`}
                                scrubLabel={componentLabels[index]}
                                scrubSensitivity={0.005}
                                step={0.01}
                                value={parameterValue}
                                onCommit={(next) => {
                                  const nextValues = [...values];
                                  nextValues[index] = next;
                                  void commitOverride(parameter, {
                                    parameterId: parameter.nodeId,
                                    name: parameter.name,
                                    type: parameter.type,
                                    kind: parameter.editorKind,
                                    value: nextValues,
                                  });
                                }}
                                onPreview={(next) => {
                                  const nextValues = [...values];
                                  nextValues[index] = next;
                                  updateLocalOverride(parameter, {
                                    parameterId: parameter.nodeId,
                                    name: parameter.name,
                                    type: parameter.type,
                                    kind: parameter.editorKind,
                                    value: nextValues,
                                  });
                                }}
                              />
                            ))}
                          </div>
                          {reset}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
      {state.status === 'ready' && state.parameters.length > 0 && (
        <div className="inspector-material-parameter-list">
          {state.parameters.map((parameter) => {
            const override = overrideFor(parameter);
            const values = effectiveValues(parameter);
            const reset = override ? (
              <button
                aria-label={`Reset ${parameter.name}`}
                className="inspector-field-reset"
                onClick={() => void commitOverride(parameter, null)}
                title="Revert to material default"
                type="button"
              >
                <RotateCcw aria-hidden="true" size={12} />
              </button>
            ) : null;

            if (parameter.editorKind === 'texture') {
              const textureValue = effectiveTexture(parameter);
              return (
                <div className="inspector-material-parameter" key={parameter.nodeId}>
                  <TexturePicker
                    allowEmpty
                    assets={assets}
                    label={parameter.name}
                    thumbnailProvider={thumbnailProvider}
                    value={textureValue}
                    onChange={(texture, asset) =>
                      void commitOverride(parameter, {
                        name: parameter.name,
                        type: parameter.type,
                        kind: parameter.editorKind,
                        texture,
                        ...(asset ? { textureReference: hostAssetReference(asset) ?? undefined } : {}),
                      })
                    }
                  />
                  {reset}
                </div>
              );
            }

            if (parameter.editorKind === 'color') {
              const rgba: Vec4 = {
                x: values[0] ?? 0,
                y: values[1] ?? 0,
                z: values[2] ?? 0,
                w: parameter.type === 'vec4' ? (values[3] ?? 1) : 1,
              };
              const colorOverride = (next: Vec4): InstanceOverride => ({
                name: parameter.name,
                type: parameter.type,
                kind: parameter.editorKind,
                value: parameter.type === 'vec4' ? [next.x, next.y, next.z, next.w] : [next.x, next.y, next.z],
              });
              return (
                <div className="inspector-material-parameter" key={parameter.nodeId}>
                  <span className="inspector-property-label">{parameter.name}</span>
                  <UiColorControl
                    allowAlpha={parameter.type === 'vec4'}
                    label={parameter.name}
                    value={rgba}
                    onPreview={(next) => updateLocalOverride(parameter, colorOverride(next))}
                    onCommit={(next) => void commitOverride(parameter, colorOverride(next))}
                  />
                  {reset}
                </div>
              );
            }

            if (parameter.editorKind === 'scalar') {
              const nextOverride = (next: number): InstanceOverride => ({
                name: parameter.name,
                type: parameter.type,
                kind: parameter.editorKind,
                value: [clampParameterValue(next, parameter.range)],
              });
              const scalarValue = clampParameterValue(values[0] ?? 0, parameter.range);
              return (
                <div className="inspector-material-parameter" key={parameter.nodeId}>
                  <div className="inspector-material-scalar-control">
                    <NumberControl
                      field={numericField(parameter.name, parameter.range)}
                      value={scalarValue}
                      onPreview={(next) => updateLocalOverride(parameter, nextOverride(next))}
                      onCommit={(next) => void commitOverride(parameter, nextOverride(next))}
                    />
                    {parameter.range && (
                      <UiSlider
                        aria-label={`${parameter.name} slider`}
                        min={parameter.range.min}
                        max={parameter.range.max}
                        step={Math.max(0.001, (parameter.range.max - parameter.range.min) / 100)}
                        value={scalarValue}
                        onValueChange={(next) => void commitOverride(parameter, nextOverride(next))}
                      />
                    )}
                  </div>
                  {reset}
                </div>
              );
            }

            return (
              <div className="inspector-material-parameter" key={parameter.nodeId}>
                <span className="inspector-property-label" title={`${parameter.name} (${parameter.type})`}>
                  {parameter.name}
                </span>
                <div className="inspector-material-parameter-values">
                  {values.map((parameterValue, index) => (
                    <UiNumericInput
                      ariaLabel={`${parameter.name} ${componentLabels[index]}`}
                      key={index}
                      precision={3}
                      scrubClassName={`axis-${componentLabels[index].toLocaleLowerCase()}`}
                      scrubLabel={componentLabels[index]}
                      scrubSensitivity={0.005}
                      step={0.01}
                      value={parameterValue}
                      onCommit={(next) => {
                        const nextValues = [...values];
                        nextValues[index] = next;
                        void commitOverride(parameter, {
                          name: parameter.name,
                          type: parameter.type,
                          kind: parameter.editorKind,
                          value: nextValues,
                        });
                      }}
                      onPreview={(next) => {
                        const nextValues = [...values];
                        nextValues[index] = next;
                        updateLocalOverride(parameter, {
                          name: parameter.name,
                          type: parameter.type,
                          kind: parameter.editorKind,
                          value: nextValues,
                        });
                      }}
                    />
                  ))}
                </div>
                {reset}
              </div>
            );
          })}
        </div>
      )}
      {mutationError && (
        <p className="inspector-subsection-error" role="alert">
          <TriangleAlert aria-hidden="true" size={10} />
          <span>{mutationError}</span>
        </p>
      )}
      {state.status === 'ready' && state.parameters.length === 0 && state.functionSlots.length === 0 && (
        <p className="inspector-subsection-empty">No exported parameters.</p>
      )}
      {state.status === 'custom' && (
        <p className="inspector-subsection-empty">Custom shader parameters are reflected during asset cook.</p>
      )}
      {state.status === 'error' && (
        <p className="inspector-subsection-empty">Parameter metadata is unavailable for this material.</p>
      )}
      {state.status === 'loading' && <p className="inspector-subsection-empty">Reading material parameters…</p>}
    </section>
  );
}
