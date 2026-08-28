import { useState } from 'react';

import { AssetPreviewPanel, AssetPreviewPlaceholder } from '../assetPreview/AssetPreviewPanel';
import type { EditorDocument } from '../editors/editorTypes';
import { MaterialGraphEditor } from '../material/MaterialGraphEditor';
import type { MaterialGraph } from '../material/materialGraphTypes';
import { UiNodeCard } from '../ui';

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

const toHex = (value: readonly number[]) =>
  `#${value
    .slice(0, 3)
    .map((channel) =>
      Math.round(Math.min(1, Math.max(0, channel)) * 255)
        .toString(16)
        .padStart(2, '0'),
    )
    .join('')}`;

export function UiLabMaterialNodeCard() {
  const [color, setColor] = useState([0.42, 0.24, 0.12, 1]);
  const [parameter, setParameter] = useState(true);
  const [parameterName, setParameterName] = useState('Base Color');
  const channels = ['R', 'G', 'B', 'A'] as const;

  return (
    <UiNodeCard badge="vec4" className="ui-lab-material-node" heading="Color (RGBA)">
      <div className="material-node-pins">
        <div className="material-node-inputs" />
        <div className="material-node-outputs">
          <button className="material-pin output" disabled type="button">
            <span>RGBA</span>
            <i />
          </button>
        </div>
      </div>
      <div className="material-node-color-value">
        <input
          aria-label="Material node color picker"
          className="material-node-color-swatch"
          type="color"
          value={toHex(color)}
          onChange={(event) => {
            const hex = event.target.value;
            setColor((current) => [
              Number.parseInt(hex.slice(1, 3), 16) / 255,
              Number.parseInt(hex.slice(3, 5), 16) / 255,
              Number.parseInt(hex.slice(5, 7), 16) / 255,
              current[3],
            ]);
          }}
        />
        <div className="material-node-color-components">
          {channels.map((channel, index) => (
            <label key={channel}>
              {channel}
              <input
                aria-label={`Material node ${channel}`}
                max={1}
                min={0}
                step="0.01"
                type="number"
                value={color[index]}
                onChange={(event) =>
                  setColor((current) => {
                    const next = [...current];
                    next[index] = Number(event.target.value);
                    return next;
                  })
                }
              />
            </label>
          ))}
        </div>
      </div>
      <label className="material-node-parameter-toggle">
        <input checked={parameter} type="checkbox" onChange={(event) => setParameter(event.target.checked)} />
        Parameter
        {parameter && (
          <input
            aria-label="Material node parameter name"
            value={parameterName}
            onChange={(event) => setParameterName(event.target.value)}
          />
        )}
      </label>
    </UiNodeCard>
  );
}

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
