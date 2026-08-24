import { useEffect, useMemo, useState } from 'react';

import { materialEditorParameters } from '../material/materialCompiler';
import {
  materialGraphFromAsset,
  type MaterialAssetJson,
  type MaterialGraphNode,
  type MaterialGraphValueType,
} from '../material/materialGraphTypes';

import './inspectorPolish.css';

type MaterialParameterAsset = {
  id: string;
  guid?: string;
  name: string;
  path: string;
  kind: string;
  scope?: 'builtin' | 'project' | 'user' | 'organization' | 'procedural';
};

type DisplayParameter = {
  nodeId: string;
  name: string;
  type: MaterialGraphValueType;
  values: number[];
};

type ParameterState =
  | { status: 'idle'; parameters: DisplayParameter[] }
  | { status: 'loading'; parameters: DisplayParameter[] }
  | { status: 'ready'; parameters: DisplayParameter[] }
  | { status: 'custom'; parameters: DisplayParameter[] }
  | { status: 'error'; parameters: DisplayParameter[] };

const emptyState: ParameterState = { status: 'idle', parameters: [] };
const componentLabels = ['X', 'Y', 'Z', 'W'];

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

  try {
    const snapshot = await window.arc.projects.snapshot();
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

const formatParameter = (value: number) => {
  const absolute = Math.abs(value);
  if (absolute > 0 && absolute < 0.001) return value.toExponential(2);
  if (absolute >= 10000) return value.toFixed(0);
  return value.toFixed(3);
};

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
  const materialPath = selected?.path ?? (referenceMode === 'path' && /\.arcmat$/i.test(value) ? value : '');
  const materialScope = selected?.scope === 'builtin' ? 'builtin' : 'project';
  const procedural = selected?.scope === 'procedural';
  const [state, setState] = useState<ParameterState>(emptyState);

  useEffect(() => {
    let active = true;
    if (!value || mixed || !materialPath || procedural) {
      setState(emptyState);
      return () => {
        active = false;
      };
    }

    setState({ status: 'loading', parameters: [] });
    void (async () => {
      try {
        const resolvedMaterialPath = await projectRelativeMaterialPath(materialPath, materialScope);
        const file = await window.arc.projects.readText(resolvedMaterialPath, materialScope);
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
          };
        });
        setState({ status: 'ready', parameters });
      } catch {
        if (active) setState({ status: 'error', parameters: [] });
      }
    })();

    return () => {
      active = false;
    };
  }, [materialPath, materialScope, mixed, procedural, value]);

  if (!value || mixed || !materialPath || procedural) return null;

  const summary =
    state.status === 'loading'
      ? 'Loading…'
      : state.status === 'ready'
        ? `${state.parameters.length} exposed`
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
          {state.parameters.map((parameter) => (
            <div className="inspector-material-parameter" key={parameter.nodeId}>
              <span className="inspector-property-label" title={`${parameter.name} (${parameter.type})`}>
                {parameter.name}
              </span>
              <div className="inspector-material-parameter-values">
                {parameter.values.map((parameterValue, index) => (
                  <output
                    aria-label={`${parameter.name}${parameter.values.length > 1 ? ` ${componentLabels[index]}` : ''}`}
                    key={index}
                    title="Shared material default. Open the material to edit it."
                  >
                    {parameter.values.length > 1 && <i>{componentLabels[index]}</i>}
                    <span>{formatParameter(parameterValue)}</span>
                  </output>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
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
