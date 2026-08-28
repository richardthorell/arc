import { AssetPreviewPanel, AssetPreviewPlaceholder } from '../assetPreview/AssetPreviewPanel';
import type { EditorDocument } from '../editors/editorTypes';
import { MaterialGraphEditor } from '../material/MaterialGraphEditor';
import type { MaterialGraph } from '../material/materialGraphTypes';

import '../material/materialEditor.css';
import '../material/materialWorkspace.css';

const materialDocument: EditorDocument = {
  id: 'ui-lab-material',
  kind: 'material',
  title: 'M_Wood_Logs',
  path: 'Assets/Materials/M_Wood_Logs.arcmat',
  assetGuid: '00000000-0000-0000-0000-000000000001',
  dirty: false,
  readOnly: false,
};

const materialGraph: MaterialGraph = {
  version: 1,
  nodes: [
    {
      id: 'base-color',
      type: 'colorRgb',
      position: [80, 120],
      values: { value: [0.42, 0.24, 0.12] },
      parameter: { exposed: true, name: 'Base Color' },
    },
    {
      id: 'roughness',
      type: 'constant',
      position: [80, 300],
      values: { value: 0.45 },
      parameter: { exposed: true, name: 'Roughness' },
    },
    {
      id: 'output',
      type: 'output',
      position: [460, 180],
      values: {},
    },
  ],
  connections: [
    {
      id: 'base-color-output',
      from: { nodeId: 'base-color', pin: 'value' },
      to: { nodeId: 'output', pin: 'baseColor' },
    },
    {
      id: 'roughness-output',
      from: { nodeId: 'roughness', pin: 'value' },
      to: { nodeId: 'output', pin: 'roughness' },
    },
  ],
  viewport: { x: 0, y: 0, zoom: 1 },
};

export function UiLabMaterialPreview() {
  return (
    <AssetPreviewPanel
      title="Material Preview"
      subtitle="Native renderer"
      metadata={[
        { label: 'Mesh', value: 'Sphere' },
        { label: 'Environment', value: 'Studio' },
      ]}
    >
      <AssetPreviewPlaceholder
        label="Material preview"
        description="UI Lab uses the production preview panel without requiring the native renderer."
      />
    </AssetPreviewPanel>
  );
}

export function UiLabMaterialParameters() {
  const baseColor = [
    ['X', 0.42],
    ['Y', 0.24],
    ['Z', 0.12],
  ] as const;

  return (
    <section className="material-parameters-panel" aria-label="Material parameters">
      <header>
        <div>
          <strong>Parameters</strong>
          <span>2 exposed</span>
        </div>
      </header>
      <div className="material-parameter-list">
        <label className="material-parameter">
          <span>
            <strong>Base Color</strong>
            <small>vec3</small>
          </span>
          <div>
            {baseColor.map(([label, value]) => (
              <span className="material-parameter-component" key={label}>
                <i>{label}</i>
                <input aria-label={`Base Color ${label}`} defaultValue={value} step="0.01" type="number" />
              </span>
            ))}
          </div>
        </label>
        <label className="material-parameter">
          <span>
            <strong>Roughness</strong>
            <small>float</small>
          </span>
          <div>
            <span className="material-parameter-component">
              <input aria-label="Roughness" defaultValue={0.45} max={1} min={0} step="0.01" type="number" />
            </span>
          </div>
        </label>
      </div>
    </section>
  );
}

export function UiLabMaterialGraph() {
  return <MaterialGraphEditor document={materialDocument} graph={materialGraph} />;
}
