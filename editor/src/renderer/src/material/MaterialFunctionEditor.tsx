import { useMemo } from 'react';

import type { EditorDocument } from '../editors/editorTypes';
import { UiButton, UiPanelCard, UiPanelCardRow, UiSelect } from '../ui';
import { MaterialGraphEditor } from './MaterialGraphEditor';
import {
  redoMaterialFunctionGraph,
  replaceMaterialFunctionAsset,
  replaceMaterialFunctionGraph,
  replaceMaterialFunctionViewport,
  undoMaterialFunctionGraph,
  useMaterialFunctionDocumentState,
} from './materialFunctionDocumentState';
import {
  cloneMaterialGraph,
  materialGraphId,
  type MaterialFunctionPin,
  type MaterialGraph,
} from './materialGraphTypes';

const typeOptions = [
  { value: 'float', label: 'Float' },
  { value: 'vec2', label: 'Vector 2' },
  { value: 'vec3', label: 'Vector 3' },
  { value: 'vec4', label: 'Vector 4' },
];

const defaultForType = (type: MaterialFunctionPin['type']) => {
  if (type === 'float') return 0;
  return Array.from({ length: type === 'vec2' ? 2 : type === 'vec3' ? 3 : 4 }, () => 0);
};

const syncBoundaryNodes = (
  graph: MaterialGraph,
  inputs: MaterialFunctionPin[],
  outputs: MaterialFunctionPin[],
): MaterialGraph => {
  const next = cloneMaterialGraph(graph);
  const inputIds = new Set(inputs.map((pin) => pin.id));
  const outputIds = new Set(outputs.map((pin) => pin.id));

  next.nodes = next.nodes.filter(
    (node) => node.type !== 'functionInput' || inputIds.has(String(node.values.input ?? '')),
  );

  for (const [index, input] of inputs.entries()) {
    const existing = next.nodes.find((node) => node.type === 'functionInput' && node.values.input === input.id);
    if (existing) {
      existing.values = { input: input.id, name: input.name, valueType: input.type };
    } else {
      next.nodes.push({
        id: `function-input-${input.id}`,
        type: 'functionInput',
        position: [80, 100 + index * 120],
        values: { input: input.id, name: input.name, valueType: input.type },
      });
    }
  }

  let output = next.nodes.find((node) => node.type === 'functionOutput');
  if (!output) {
    output = {
      id: 'function-output',
      type: 'functionOutput',
      position: [560, 120],
      values: { pins: outputs },
    };
    next.nodes.push(output);
  } else {
    output.values = { ...output.values, pins: outputs };
  }

  next.connections = next.connections.filter((connection) => {
    const from = next.nodes.find((node) => node.id === connection.from.nodeId);
    const to = next.nodes.find((node) => node.id === connection.to.nodeId);
    if (!from || !to) return false;
    if (from.type === 'functionInput' && !inputIds.has(String(from.values.input ?? ''))) return false;
    if (to.type === 'functionOutput' && !outputIds.has(connection.to.pin)) return false;
    return true;
  });
  return next;
};

const pinId = (prefix: string) => `${prefix}_${materialGraphId('pin').replaceAll('-', '_')}`;

export function MaterialFunctionEditor({ document }: { document: EditorDocument }) {
  const state = useMaterialFunctionDocumentState(document);

  const updateSignature = (inputs: MaterialFunctionPin[], outputs: MaterialFunctionPin[]) => {
    replaceMaterialFunctionAsset(document, (asset) => ({ ...asset, inputs, outputs }));
    replaceMaterialFunctionGraph(document, syncBoundaryNodes(state.graph, inputs, outputs), {
      message: 'Updated Material Function signature',
    });
  };

  const updatePin = (direction: 'inputs' | 'outputs', id: string, patch: Partial<MaterialFunctionPin>) => {
    const inputs =
      direction === 'inputs'
        ? state.asset.inputs.map((pin) => (pin.id === id ? { ...pin, ...patch } : pin))
        : state.asset.inputs;
    const outputs =
      direction === 'outputs'
        ? state.asset.outputs.map((pin) => (pin.id === id ? { ...pin, ...patch } : pin))
        : state.asset.outputs;
    updateSignature(inputs, outputs);
  };

  const removePin = (direction: 'inputs' | 'outputs', id: string) => {
    const inputs = direction === 'inputs' ? state.asset.inputs.filter((pin) => pin.id !== id) : state.asset.inputs;
    const outputs = direction === 'outputs' ? state.asset.outputs.filter((pin) => pin.id !== id) : state.asset.outputs;
    if (outputs.length === 0) return;
    updateSignature(inputs, outputs);
  };

  const signature = useMemo(
    () => [
      { title: 'Inputs', direction: 'inputs' as const, pins: state.asset.inputs },
      { title: 'Outputs', direction: 'outputs' as const, pins: state.asset.outputs },
    ],
    [state.asset.inputs, state.asset.outputs],
  );

  return (
    <section className="material-editor" style={{ gridTemplateColumns: 'minmax(520px, 1fr) 5px 420px' }}>
      <div className="material-editor-graph-region">
        <MaterialGraphEditor
          document={document}
          graph={state.graph}
          loaded={state.loaded}
          onGraphChange={(graph, options) => replaceMaterialFunctionGraph(document, graph, options)}
          onViewportChange={(viewport) => replaceMaterialFunctionViewport(document, viewport)}
          onUndo={() => undoMaterialFunctionGraph(document)}
          onRedo={() => redoMaterialFunctionGraph(document)}
        />
        {state.message && (
          <div className="material-editor-message" role="status">
            {state.message}
          </div>
        )}
      </div>

      <div className="material-editor-divider" aria-hidden="true" />

      <aside className="material-editor-sidebar editor-property-panel">
        <div className="material-settings-region">
          <UiPanelCard className="material-settings-card" title="Material Function">
            <UiPanelCardRow label="Name">
              <input
                disabled={document.readOnly}
                value={state.asset.name}
                onChange={(event) =>
                  replaceMaterialFunctionAsset(document, (asset) => ({ ...asset, name: event.target.value }))
                }
              />
            </UiPanelCardRow>
            <UiPanelCardRow label="Description">
              <input
                disabled={document.readOnly}
                value={state.asset.description ?? ''}
                onChange={(event) =>
                  replaceMaterialFunctionAsset(document, (asset) => ({ ...asset, description: event.target.value }))
                }
              />
            </UiPanelCardRow>
          </UiPanelCard>

          {signature.map(({ title, direction, pins }) => (
            <UiPanelCard className="material-settings-card" key={direction} title={title}>
              {pins.map((pin) => (
                <div className="material-settings-list" key={pin.id}>
                  <UiPanelCardRow label={pin.id}>
                    <input
                      aria-label={`${title} pin name`}
                      disabled={document.readOnly}
                      value={pin.name}
                      onChange={(event) => updatePin(direction, pin.id, { name: event.target.value })}
                    />
                  </UiPanelCardRow>
                  <UiPanelCardRow label="Type">
                    <UiSelect
                      ariaLabel={`${title} pin type`}
                      disabled={document.readOnly}
                      options={typeOptions}
                      value={pin.type}
                      onValueChange={(value) => {
                        const type = value as MaterialFunctionPin['type'];
                        updatePin(direction, pin.id, {
                          type,
                          ...(direction === 'inputs' && pin.default !== undefined
                            ? { default: defaultForType(type) }
                            : {}),
                        });
                      }}
                    />
                  </UiPanelCardRow>
                  {direction === 'inputs' && (
                    <UiPanelCardRow label="Optional">
                      <input
                        aria-label="Input has default"
                        checked={pin.default !== undefined}
                        disabled={document.readOnly}
                        type="checkbox"
                        onChange={(event) =>
                          updatePin(direction, pin.id, {
                            default: event.target.checked ? defaultForType(pin.type) : undefined,
                          })
                        }
                      />
                    </UiPanelCardRow>
                  )}
                  <UiPanelCardRow label="">
                    <UiButton
                      disabled={document.readOnly || (direction === 'outputs' && pins.length === 1)}
                      onClick={() => removePin(direction, pin.id)}
                    >
                      Remove
                    </UiButton>
                  </UiPanelCardRow>
                </div>
              ))}
              <UiPanelCardRow label="">
                <UiButton
                  disabled={document.readOnly}
                  onClick={() => {
                    const id = pinId(direction === 'inputs' ? 'input' : 'output');
                    const next: MaterialFunctionPin = {
                      id,
                      name: direction === 'inputs' ? 'Input' : 'Output',
                      type: 'float',
                    };
                    updateSignature(
                      direction === 'inputs' ? [...state.asset.inputs, next] : state.asset.inputs,
                      direction === 'outputs' ? [...state.asset.outputs, next] : state.asset.outputs,
                    );
                  }}
                >
                  Add {direction === 'inputs' ? 'Input' : 'Output'}
                </UiButton>
              </UiPanelCardRow>
            </UiPanelCard>
          ))}
        </div>
      </aside>
    </section>
  );
}
