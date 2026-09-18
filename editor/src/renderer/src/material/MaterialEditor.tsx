import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent, PointerEvent } from 'react';
import { AlertCircle, CheckCircle2, Code2, Lock } from 'lucide-react';

import { AssetPreviewPanel, AssetPreviewPlaceholder } from '../assetPreview/AssetPreviewPanel';
import { AssetPreviewViewport } from '../assetPreview/AssetPreviewViewport';
import type { EditorDocument } from '../editors/editorTypes';
import { materialEditorParameters } from './materialCompiler';
import { replaceMaterialGraph, useMaterialDocumentState } from './materialDocumentState';
import { MaterialGraphWithInteractions } from './MaterialGraphInteractions';
import { cloneMaterialGraph, type MaterialGraphNode } from './materialGraphTypes';
import './materialCustomShader.css';
import './materialEditor.css';
import './materialWorkspace.css';

export const defaultMaterialSidebarWidth = 560;
export const minimumMaterialSidebarWidth = 320;
export const maximumMaterialSidebarWidth = 640;
export const minimumMaterialGraphWidth = 520;
export const materialEditorDividerWidth = 5;

export function clampMaterialSidebarWidth(containerWidth: number, requestedWidth: number): number {
  const availableWidth = Math.max(
    minimumMaterialSidebarWidth,
    containerWidth - minimumMaterialGraphWidth - materialEditorDividerWidth,
  );
  const maximumWidth = Math.min(maximumMaterialSidebarWidth, availableWidth);
  return Math.round(Math.min(maximumWidth, Math.max(minimumMaterialSidebarWidth, requestedWidth)));
}

const parameterValue = (node: MaterialGraphNode): number[] => {
  if (typeof node.values.value === 'number') return [node.values.value];
  if (Array.isArray(node.values.value))
    return node.values.value.map((value) => (typeof value === 'number' ? value : 0));
  return [];
};

const componentLabels = ['X', 'Y', 'Z', 'W'];
const materialPreviewMeshes = ['sphere', 'cube', 'pill'] as const;
type MaterialPreviewMesh = (typeof materialPreviewMeshes)[number];

type SidebarResize = {
  pointerId: number;
  startX: number;
  startWidth: number;
};

export function MaterialEditor({ document }: { document: EditorDocument }) {
  const state = useMaterialDocumentState(document);
  const customShader = typeof state.asset.shaderPath === 'string' ? state.asset.shaderPath.trim() : '';
  const parameters = customShader ? [] : materialEditorParameters(state.graph);
  const errors = state.compilation.diagnostics.filter((diagnostic) => diagnostic.severity === 'error');
  const warnings = state.compilation.diagnostics.filter((diagnostic) => diagnostic.severity === 'warning');
  const editorRef = useRef<HTMLElement | null>(null);
  const sidebarResizeRef = useRef<SidebarResize | null>(null);
  const [sidebarWidth, setSidebarWidth] = useState(defaultMaterialSidebarWidth);
  const [previewMesh, setPreviewMesh] = useState<MaterialPreviewMesh>('sphere');
  const previewAssetGuid =
    document.assetGuid && previewMesh !== 'sphere' ? `${document.assetGuid}~${previewMesh}` : document.assetGuid;
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

  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    const clampToEditor = () => {
      const width = editor.getBoundingClientRect().width;
      setSidebarWidth((current) => clampMaterialSidebarWidth(width, current));
    };
    clampToEditor();
    const observer = new ResizeObserver(clampToEditor);
    observer.observe(editor);
    return () => observer.disconnect();
  }, []);

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

  const resizeSidebar = (requestedWidth: number) => {
    const containerWidth = editorRef.current?.getBoundingClientRect().width ?? window.innerWidth;
    setSidebarWidth(clampMaterialSidebarWidth(containerWidth, requestedWidth));
  };

  const onSidebarResizeStart = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    sidebarResizeRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startWidth: sidebarWidth,
    };
  };

  const onSidebarResizeMove = (event: PointerEvent<HTMLDivElement>) => {
    const resize = sidebarResizeRef.current;
    if (!resize || resize.pointerId !== event.pointerId) return;
    resizeSidebar(resize.startWidth + resize.startX - event.clientX);
  };

  const finishSidebarResize = (event: PointerEvent<HTMLDivElement>) => {
    if (sidebarResizeRef.current?.pointerId === event.pointerId) sidebarResizeRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const onSidebarResizeKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight' && event.key !== 'Home') return;
    event.preventDefault();
    if (event.key === 'Home') {
      resizeSidebar(defaultMaterialSidebarWidth);
      return;
    }
    resizeSidebar(sidebarWidth + (event.key === 'ArrowLeft' ? 16 : -16));
  };

  return (
    <section
      ref={editorRef}
      className="material-editor"
      style={{
        gridTemplateColumns: `minmax(${minimumMaterialGraphWidth}px, 1fr) ${materialEditorDividerWidth}px ${sidebarWidth}px`,
      }}
    >
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
        <MaterialGraphWithInteractions document={document} graph={state.graph} loaded={state.loaded} />
      )}

      <div
        className="material-editor-divider"
        role="separator"
        aria-label="Resize material preview panel"
        aria-orientation="vertical"
        aria-valuemin={minimumMaterialSidebarWidth}
        aria-valuemax={maximumMaterialSidebarWidth}
        aria-valuenow={sidebarWidth}
        tabIndex={0}
        style={{
          cursor: 'col-resize',
          touchAction: 'none',
          borderLeft: '1px solid rgba(102, 132, 146, 0.14)',
          borderRight: '1px solid rgba(102, 132, 146, 0.22)',
          background: '#0e171c',
        }}
        onPointerDown={onSidebarResizeStart}
        onPointerMove={onSidebarResizeMove}
        onPointerUp={finishSidebarResize}
        onPointerCancel={finishSidebarResize}
        onDoubleClick={() => resizeSidebar(defaultMaterialSidebarWidth)}
        onKeyDown={onSidebarResizeKeyDown}
      />

      <aside className="material-editor-sidebar editor-property-panel">
        <AssetPreviewPanel
          title="Material Preview"
          subtitle="Native renderer"
          metadata={[
            {
              label: 'Mesh',
              value: (
                <span className="material-preview-mesh-toggle" role="group" aria-label="Material preview mesh">
                  {materialPreviewMeshes.map((mesh) => (
                    <button
                      key={mesh}
                      type="button"
                      aria-pressed={previewMesh === mesh}
                      onClick={() => setPreviewMesh(mesh)}
                    >
                      {mesh[0].toUpperCase() + mesh.slice(1)}
                    </button>
                  ))}
                </span>
              ),
            },
            { label: 'Environment', value: 'Studio HDRI' },
          ]}
        >
          <AssetPreviewViewport
            kind="material"
            assetGuid={previewAssetGuid}
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
