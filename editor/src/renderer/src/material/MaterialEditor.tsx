import { AlertCircle, CheckCircle2, Code2, Lock } from 'lucide-react';

import { AssetPreviewPanel, AssetPreviewPlaceholder } from '../assetPreview/AssetPreviewPanel';
import { AssetPreviewViewport } from '../assetPreview/AssetPreviewViewport';
import type { EditorDocument } from '../editors/editorTypes';
import { MaterialGraphEditor } from './MaterialGraphEditor';
import { materialEditorParameters } from './materialCompiler';
import { replaceMaterialGraph, useMaterialDocumentState } from './materialDocumentState';
import { cloneMaterialGraph, type MaterialGraphNode } from './materialGraphTypes';
import './materialCustomShader.css';
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
  const customShader = typeof state.asset.shaderPath === 'string' ? state.asset.shaderPath.trim() : '';
  const parameters = customShader ? [] : materialEditorParameters(state.graph);
  const errors = state.compilation.diagnostics.filter((diagnostic) => diagnostic.severity === 'error');
  const warnings = state.compilation.diagnostics.filter((diagnostic) => diagnostic.severity === 'warning');
  const fallbackPreview = state.previewDataUrl ? (
    <img alt={`${document.title} material preview`} src={state.previewDataUrl} />
  ) : (
    <AssetPreviewPlaceholder
      label={state.previewLoading ? 'Rendering preview…' : 'Material preview'}
      description={
        state.previewLoading
          ? 'Generating the fallback thumbnail.'
          : customShader
            ? 'Save & Reimport to refresh preview.'
            : 'Save & Compile to render preview.'
      }
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
      {customShader ? (
        <section className="material-custom-shader">
          <Code2 size={30} />
          <div>
            <strong>Custom Material Shader</strong>
            <p>
              This material implements the Material ABI with handwritten Slang. ARC owns render-pass entry points and
              composes this evaluator into the same depth, shadow, G-buffer, forward and motion passes as graph
              materials.
            </p>
            <code>{customShader}</code>
          </div>
        </section>
      ) : (
        <MaterialGraphEditor document={document} graph={state.graph} />
      )}

      <aside className="material-editor-sidebar editor-property-panel">
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

        <section className="material-parameters-panel editor-property-section">
          <header>
            <div>
              <strong>Parameters</strong>
              <span>{customShader ? 'Reflected during cook' : `${parameters.length} exposed`}</span>
            </div>
            {document.readOnly && (
              <span className="material-readonly-badge">
                <Lock size={11} /> Read-only
              </span>
            )}
          </header>
          <div className="material-parameter-list">
            {parameters.map((parameter) => {
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
            {!parameters.length && (
              <div className="material-empty-parameters">
                {customShader
                  ? 'Custom Material Shader parameters are reflected by the material cooker during asset cook.'
                  : 'Expose a Constant or Vector node as a parameter to edit it here.'}
              </div>
            )}
          </div>
        </section>

        <section className="material-details-panel editor-property-section">
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
            <dt>Implementation</dt>
            <dd>{customShader ? 'Material Shader' : 'Material Graph'}</dd>
            <dt>{customShader ? 'Source' : 'Compiler'}</dt>
            <dd>{customShader || 'Native Material IR'}</dd>
          </dl>
          <div className="material-compile-summary">
            {customShader ? (
              <Code2 size={13} />
            ) : errors.length ? (
              <AlertCircle size={13} />
            ) : (
              <CheckCircle2 size={13} />
            )}
            <span>
              {customShader
                ? 'Validated during asset cook'
                : state.compilation.status === 'compiling'
                  ? 'Native compiler running…'
                  : errors.length
                    ? `${errors.length} error${errors.length === 1 ? '' : 's'}`
                    : warnings.length
                      ? `Compiled with ${warnings.length} warning${warnings.length === 1 ? '' : 's'}`
                      : state.compilation.succeeded
                        ? 'Native compilation succeeded'
                        : 'Awaiting native compilation'}
            </span>
          </div>
          {!customShader && (errors.length > 0 || warnings.length > 0) && (
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
