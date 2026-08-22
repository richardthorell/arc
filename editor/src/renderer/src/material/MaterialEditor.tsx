import { AlertCircle, CheckCircle2, Lock } from 'lucide-react';

import { AssetPreviewPanel, AssetPreviewPlaceholder } from '../assetPreview/AssetPreviewPanel';
import { AssetPreviewViewport } from '../assetPreview/AssetPreviewViewport';
import type { EditorDocument } from '../editors/editorTypes';
import { MaterialGraphEditor } from './MaterialGraphEditor';
import { replaceMaterialGraph, useMaterialDocumentState } from './materialDocumentState';
import { cloneMaterialGraph, type MaterialGraphNode } from './materialGraphTypes';
import './materialEditor.css';
import './materialWorkspace.css';

const parameterValue = (node: MaterialGraphNode): number[] => {
  if (typeof node.values.value === 'number') return [node.values.value];
  if (Array.isArray(node.values.value))
    return node.values.value.map((value) => (typeof value === 'number' ? value : 0));
  return [];
};

const componentLabels = ['X', 'Y', 'Z', 'W'];

export function MaterialEditor({ document }: { document: EditorDocument }) {
  const state = useMaterialDocumentState(document);
  const errors = state.compilation.diagnostics.filter((diagnostic) => diagnostic.severity === 'error');
  const warnings = state.compilation.diagnostics.filter((diagnostic) => diagnostic.severity === 'warning');
  const fallbackPreview = state.previewDataUrl ? (
    <img alt={`${document.title} material preview`} src={state.previewDataUrl} />
  ) : (
    <AssetPreviewPlaceholder
      label={state.previewLoading ? 'Rendering preview…' : 'Material preview'}
      description={state.previewLoading ? 'Generating the fallback thumbnail.' : 'Save & Compile to render preview.'}
    />
  );

  const setParameterComponent = (nodeId: string, component: number, value: number) => {
    const next = cloneMaterialGraph(state.graph);
    const node = next.nodes.find((candidate) => candidate.id === nodeId);
    if (!node) return;
    if (typeof node.values.value === 'number') node.values.value = value;
    else {
      const values = Array.isArray(node.values.value) ? [...node.values.value] : [0];
      values[component] = value;
      node.values.value = values;
    }
    replaceMaterialGraph(document, next);
  };

  return (
    <section className="material-editor">
      <MaterialGraphEditor document={document} graph={state.graph} />

      <aside className="material-editor-sidebar">
        <AssetPreviewPanel
          title="Material Preview"
          subtitle="Native renderer"
          metadata={[
            { label: 'Mesh', value: 'Sphere' },
            { label: 'Environment', value: 'Studio' },
          ]}
        >
          <AssetPreviewViewport
            kind="material"
            assetGuid={document.assetGuid}
            label={`${document.title} material preview viewport`}
            fallback={fallbackPreview}
          />
        </AssetPreviewPanel>

        <section className="material-parameters-panel">
          <header>
            <div>
              <strong>Parameters</strong>
              <span>{state.compilation.ir.parameters.length} exposed</span>
            </div>
            {document.readOnly && (
              <span className="material-readonly-badge">
                <Lock size={11} /> Read-only
              </span>
            )}
          </header>
          <div className="material-parameter-list">
            {state.compilation.ir.parameters.map((parameter) => {
              const node = state.graph.nodes.find((candidate) => candidate.id === parameter.nodeId);
              if (!node) return null;
              const values = parameterValue(node);
              return (
                <label className="material-parameter" key={parameter.nodeId}>
                  <span>
                    <strong>{parameter.name}</strong>
                    <small>{parameter.type}</small>
                  </span>
                  <div>
                    {values.map((value, index) => (
                      <span className="material-parameter-component" key={index}>
                        {values.length > 1 && <i>{componentLabels[index]}</i>}
                        <input
                          disabled={document.readOnly}
                          type="number"
                          step="0.01"
                          value={value}
                          onChange={(event) =>
                            setParameterComponent(parameter.nodeId, index, Number(event.target.value))
                          }
                        />
                      </span>
                    ))}
                  </div>
                </label>
              );
            })}
            {!state.compilation.ir.parameters.length && (
              <div className="material-empty-parameters">
                Expose a Constant or Vector node as a parameter to edit it here.
              </div>
            )}
          </div>
        </section>

        <section className="material-details-panel">
          <header>
            <strong>Material</strong>
            <span>{state.asset.name ?? document.title}</span>
          </header>
          <dl>
            <dt>Domain</dt>
            <dd>{String(state.asset.domain ?? 'surface')}</dd>
            <dt>Blend</dt>
            <dd>{String(state.asset.blendMode ?? 'opaque')}</dd>
            <dt>Shading</dt>
            <dd>{String(state.asset.shadingModel ?? 'standard')}</dd>
            <dt>Shader</dt>
            <dd>{String(state.asset.shader ?? 'arc/default_phong')}</dd>
            <dt>IR</dt>
            <dd>{state.compilation.ir.expressions.length} expressions</dd>
          </dl>
          <div className="material-compile-summary">
            {errors.length ? <AlertCircle size={13} /> : <CheckCircle2 size={13} />}
            <span>
              {errors.length
                ? `${errors.length} error${errors.length === 1 ? '' : 's'}`
                : warnings.length
                  ? `Compiled with ${warnings.length} warning${warnings.length === 1 ? '' : 's'}`
                  : 'Graph compiles cleanly'}
            </span>
          </div>
          {(errors.length > 0 || warnings.length > 0) && (
            <div className="material-diagnostics">
              {[...errors, ...warnings].map((diagnostic, index) => (
                <p className={diagnostic.severity} key={`${diagnostic.nodeId ?? 'graph'}-${index}`}>
                  {diagnostic.message}
                </p>
              ))}
            </div>
          )}
        </section>
      </aside>

      {state.message && <div className="material-editor-message">{state.message}</div>}
    </section>
  );
}
