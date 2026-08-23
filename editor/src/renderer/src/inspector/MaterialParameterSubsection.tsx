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
  value,
}: {
  assets: ReadonlyArray<MaterialParameterAsset>;
  mixed?: boolean;
  value: string;
}) {
  const selected = useMemo(
    () => assets.find((asset) => asset.kind === 'material' && asset.path === value),
    [assets, value],
  );
  const [state, setState] = useState<ParameterState>(emptyState);

  useEffect(() => {
    let active = true;
    if (!value || mixed || !selected || selected.scope === 'procedural') {
      setState(emptyState);
      return () => {
        active = false;
      };
    }

    setState({ status: 'loading', parameters: [] });
    void (async () => {
      try {
        const file = await window.arc.projects.readText(selected.path, selected.scope === 'builtin' ? 'builtin' : 'project');
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
  }, [mixed, selected, value]);

  if (!value || mixed || !selected || selected.scope === 'procedural') return null;

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
