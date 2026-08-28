import { useEffect, useMemo, useState } from 'react';
import { RotateCcw } from 'lucide-react';

import { materialEditorParameters, type MaterialEditorParameterKind } from '../material/materialCompiler';
import {
  materialGraphFromAsset,
  type MaterialAssetJson,
  type MaterialGraphNode,
  type MaterialGraphValueType,
} from '../material/materialGraphTypes';
import type { HostEntityId, HostResponse, Vec4 } from './inspectorTypes';
import { ColorControl, NumberControl, NumericInput } from './InspectorControls';

import './inspectorPolish.css';

type MaterialParameterAsset = {
  id: string;
  guid?: string;
  typeId?: string;
  name: string;
  path: string;
  kind: string;
  status: 'unknown' | 'queued' | 'ready' | 'dirty' | 'stale' | 'importing' | 'failed' | 'missing';
  scope?: 'builtin' | 'project' | 'user' | 'organization' | 'procedural';
  readOnly?: boolean;
};

type InstanceOverride = {
  name: string;
  type: MaterialGraphValueType;
  kind: MaterialEditorParameterKind;
  value?: number[];
  texture?: string;
};

type DisplayParameter = {
  nodeId: string;
  name: string;
  type: MaterialGraphValueType;
  editorKind: MaterialEditorParameterKind;
  values: number[];
  texture: string;
};

type ParameterState =
  | { status: 'idle'; parameters: DisplayParameter[] }
  | { status: 'loading'; parameters: DisplayParameter[] }
  | { status: 'ready'; parameters: DisplayParameter[] }
  | { status: 'custom'; parameters: DisplayParameter[] }
  | { status: 'error'; parameters: DisplayParameter[] };

type SelectedMaterialSnapshot = {
  entity: HostEntityId;
  selectionCount?: number;
  meshRenderer?: { materialName?: string };
};

const emptyState: ParameterState = { status: 'idle', parameters: [] };
const componentLabels = ['X', 'Y', 'Z', 'W'];
const instanceMarker = '__arc_instance_overrides__';
const parameterCommandPrefix = '__arc_primitive_parameter__/__arc_material_parameter__';

const normalizePath = (value: string) =>
  value
    .trim()
    .replaceAll('\\', '/')
    .replace(/\/+/g, '/')
    .replace(/^\.\//, '')
    .replace(/^\/|\/$/g, '');

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

const overridesFromMaterialName = (name: string | undefined): InstanceOverride[] => {
  const marker = name?.indexOf(instanceMarker) ?? -1;
  if (!name || marker < 0) return [];
  try {
    const parsed = JSON.parse(hexToText(name.slice(marker + instanceMarker.length)) || '[]');
    return Array.isArray(parsed) ? (parsed as InstanceOverride[]) : [];
  } catch {
    return [];
  }
};

const numericField = (label: string) => ({ label, precision: 3, step: 0.01, scrubSensitivity: 0.005 });

export function MaterialParameterSubsection({
  assets,
  mixed = false,
  referenceMode = 'path',
  value,
}: {
  assets: ReadonlyArray<MaterialParameterAsset>;
  mixed?: boolean;
  referenceMode?: 'path' | 'guid';
  value: string;
}) {
  const selected = useMemo(
    () =>
      assets.find(
        (asset) =>
          asset.kind === 'material' &&
          (referenceMode === 'guid' ? (asset.guid || asset.id) === value : asset.path === value),
      ),
    [assets, referenceMode, value],
  );
  const textureAssets = useMemo(
    () => assets.filter((asset) => asset.kind === 'texture' || asset.kind === 'environment'),
    [assets],
  );
  const materialPath = selected?.path ?? (referenceMode === 'path' && /\.arcmat$/i.test(value) ? value : '');
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

    setState({ status: 'loading', parameters: [] });
    void (async () => {
      try {
        const resolvedMaterialPath = await projectRelativeMaterialPath(materialPath, materialScope);
        const [file, selection] = await Promise.all([
          window.arc.projects.readText(resolvedMaterialPath, materialScope),
          window.arc.host?.query('entity.selected') as Promise<HostResponse<SelectedMaterialSnapshot>> | undefined,
        ]);
        if (!active) return;
        const asset = JSON.parse(file.text) as MaterialAssetJson;
        const customShader = typeof asset.shaderPath === 'string' ? asset.shaderPath.trim() : '';
        if (customShader) {
          setState({ status: 'custom', parameters: [] });
          return;
        }

        const graph = materialGraphFromAsset(asset);
        const parameters = materialEditorParameters(graph).map((parameter) => {
          const node = graph.nodes.find((candidate) => candidate.id === parameter.nodeId);
          return {
            ...parameter,
            values: node ? parameterValues(node) : [],
            texture: node ? parameterTexture(node) : '',
          };
        });
        setOverrides(
          selection?.succeeded ? overridesFromMaterialName(selection.payload?.meshRenderer?.materialName) : [],
        );
        setState({ status: 'ready', parameters });
      } catch {
        if (active) setState({ status: 'error', parameters: [] });
      }
    })();

    return () => {
      active = false;
    };
  }, [materialPath, materialScope, mixed, procedural, value]);

  const overrideFor = (parameter: DisplayParameter) => overrides.find((entry) => entry.name === parameter.name);
  const effectiveValues = (parameter: DisplayParameter) => overrideFor(parameter)?.value ?? parameter.values;
  const effectiveTexture = (parameter: DisplayParameter) => overrideFor(parameter)?.texture ?? parameter.texture;

  const updateLocalOverride = (parameter: DisplayParameter, next: InstanceOverride | null) => {
    setOverrides((current) => {
      const filtered = current.filter((entry) => entry.name !== parameter.name);
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
      const payload = next ?? { name: parameter.name, type: parameter.type, kind: parameter.editorKind, reset: true };
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
              const hasTextureAsset = textureAssets.some((asset) => asset.path === textureValue);
              return (
                <div className="inspector-material-parameter" key={parameter.nodeId}>
                  <label className="inspector-property inspector-asset-property">
                    <span className="inspector-property-label">{parameter.name}</span>
                    <select
                      aria-label={`Choose ${parameter.name} texture`}
                      className="inspector-select"
                      value={textureValue}
                      onChange={(event) =>
                        void commitOverride(parameter, {
                          name: parameter.name,
                          type: parameter.type,
                          kind: parameter.editorKind,
                          texture: event.target.value,
                        })
                      }
                    >
                      <option value="">None</option>
                      {textureValue && !hasTextureAsset && <option value={textureValue}>{textureValue}</option>}
                      {textureAssets.map((asset) => (
                        <option key={asset.guid || asset.id} value={asset.path}>
                          {asset.name}
                        </option>
                      ))}
                    </select>
                  </label>
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
                  <ColorControl
                    label={parameter.name}
                    showAlpha={parameter.type === 'vec4'}
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
                value: [next],
              });
              return (
                <div className="inspector-material-parameter" key={parameter.nodeId}>
                  <NumberControl
                    field={numericField(parameter.name)}
                    value={values[0] ?? 0}
                    onPreview={(next) => updateLocalOverride(parameter, nextOverride(next))}
                    onCommit={(next) => void commitOverride(parameter, nextOverride(next))}
                  />
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
                    <NumericInput
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
      {mutationError && <p className="inspector-subsection-empty">{mutationError}</p>}
      {state.status === 'ready' && state.parameters.length === 0 && (
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
