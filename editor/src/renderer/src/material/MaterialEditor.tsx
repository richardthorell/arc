import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent, PointerEvent } from 'react';
import { Code2 } from 'lucide-react';

import { AssetPreviewPanel, AssetPreviewPlaceholder } from '../assetPreview/AssetPreviewPanel';
import { AssetPreviewViewport } from '../assetPreview/AssetPreviewViewport';
import type { EditorDocument } from '../editors/editorTypes';
import { UiPanelCard, UiSelect, UiToggleButton } from '../ui';
import { replaceMaterialSettings, useMaterialDocumentState } from './materialDocumentState';
import { MaterialGraphWithInteractions } from './MaterialGraphInteractions';
import type { MaterialBlendMode, MaterialDomain, MaterialShadingModel } from './materialGraphTypes';
import './materialCustomShader.css';
import './materialEditor.css';
import './materialWorkspace.css';

export const defaultMaterialSidebarWidth = 640;
export const minimumMaterialSidebarWidth = 320;
export const maximumMaterialSidebarWidth = 760;
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

const materialPreviewMeshes = ['sphere', 'cube', 'pill'] as const;
type MaterialPreviewMesh = (typeof materialPreviewMeshes)[number];

const materialDomainOptions = [
  { value: 'surface', label: 'Surface' },
  { value: 'terrain', label: 'Terrain' },
] as const;

const materialBlendModeOptions = [
  { value: 'opaque', label: 'Opaque' },
  { value: 'masked', label: 'Masked' },
  { value: 'blend', label: 'Translucent' },
] as const;

const materialShadingModelOptions = [
  { value: 'standard', label: 'Standard' },
  { value: 'skin', label: 'Skin' },
  { value: 'transmission', label: 'Transmission' },
  { value: 'unlit', label: 'Unlit' },
  { value: 'customLit', label: 'Custom Lit' },
] as const;

type SidebarResize = {
  pointerId: number;
  startX: number;
  startWidth: number;
};

export function MaterialEditor({ document }: { document: EditorDocument }) {
  const state = useMaterialDocumentState(document);
  const customShader = typeof state.asset.shaderPath === 'string' ? state.asset.shaderPath.trim() : '';
  const materialDomain: MaterialDomain = state.asset.domain === 'terrain' ? 'terrain' : 'surface';
  const materialBlendMode: MaterialBlendMode =
    state.asset.blendMode === 'masked' || state.asset.blendMode === 'blend' ? state.asset.blendMode : 'opaque';
  const materialShadingModel: MaterialShadingModel =
    state.asset.shadingModel === 'skin' ||
    state.asset.shadingModel === 'transmission' ||
    state.asset.shadingModel === 'unlit' ||
    state.asset.shadingModel === 'customLit'
      ? state.asset.shadingModel
      : 'standard';
  const editorRef = useRef<HTMLElement | null>(null);
  const sidebarResizeRef = useRef<SidebarResize | null>(null);
  const [sidebarWidth, setSidebarWidth] = useState(defaultMaterialSidebarWidth);
  const [previewMesh, setPreviewMesh] = useState<MaterialPreviewMesh>('sphere');
  const [previewAutoRotate, setPreviewAutoRotate] = useState(true);
  const [materialSettingsCollapsed, setMaterialSettingsCollapsed] = useState(false);
  const previewLoading =
    state.loading ||
    (!customShader && (state.compilation.status === 'idle' || state.compilation.status === 'compiling'));
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
          showHeader={false}
          metadata={[
            {
              label: 'Mesh',
              value: (
                <span className="material-preview-controls">
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
                  <button
                    className="material-preview-auto-rotate"
                    type="button"
                    aria-pressed={previewAutoRotate}
                    title="Slowly rotate the preview mesh around Y"
                    onClick={() => setPreviewAutoRotate((enabled) => !enabled)}
                  >
                    Rotate
                  </button>
                </span>
              ),
            },
            { label: 'Environment', value: 'Studio HDRI' },
          ]}
        >
          <AssetPreviewViewport
            kind="material"
            assetGuid={document.assetGuid}
            materialMesh={previewMesh}
            materialAutoRotate={previewAutoRotate}
            loading={previewLoading}
            label={`${document.title} material preview viewport`}
            fallback={fallbackPreview}
          />
        </AssetPreviewPanel>

        <div className="material-settings-region">
          <UiPanelCard
            className="material-settings-card"
            collapsed={materialSettingsCollapsed}
            contentClassName="material-settings-list"
            title="Material"
            onToggle={() => setMaterialSettingsCollapsed((collapsed) => !collapsed)}
          >
            <div className="material-setting-row">
              <span>Domain</span>
              <UiSelect
                ariaLabel="Material domain"
                disabled={document.readOnly}
                options={materialDomainOptions}
                value={materialDomain}
                onValueChange={(value) => replaceMaterialSettings(document, { domain: value as MaterialDomain })}
              />
            </div>
            <div className="material-setting-row">
              <span>Blend Mode</span>
              <UiSelect
                ariaLabel="Material blend mode"
                disabled={document.readOnly}
                options={materialBlendModeOptions}
                value={materialBlendMode}
                onValueChange={(value) => replaceMaterialSettings(document, { blendMode: value as MaterialBlendMode })}
              />
            </div>
            <div className="material-setting-row">
              <span>Shading Model</span>
              <UiSelect
                ariaLabel="Material shading model"
                disabled={document.readOnly}
                options={materialShadingModelOptions}
                value={materialShadingModel}
                onValueChange={(value) =>
                  replaceMaterialSettings(document, { shadingModel: value as MaterialShadingModel })
                }
              />
            </div>
            <div className="material-setting-row material-setting-toggle-row">
              <span>Two Sided</span>
              <UiToggleButton
                aria-label="Two sided material"
                checked={state.asset.doubleSided === true}
                disabled={document.readOnly}
                onCheckedChange={(checked) => replaceMaterialSettings(document, { doubleSided: checked })}
              />
            </div>
          </UiPanelCard>
        </div>
      </aside>

      {state.message && <div className="material-editor-message">{state.message}</div>}
    </section>
  );
}
