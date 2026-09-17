import { useEffect } from 'react';
import { Plus, Trash2 } from 'lucide-react';

import type { EditorDocument, EditorSurfaceContext } from '../editors/editorTypes';
import { UiButton } from '../ui';
import { FlowGraphEditor } from './FlowGraphEditor';
import { disposeFlowDocument, replaceFlowGraph, useFlowDocumentState } from './flowDocumentState';
import { flowGraphId, type FlowGraph, type FlowValueType } from './flowGraphTypes';
import './flowEditor.css';

const variableTypes: FlowValueType[] = [
  'bool',
  'int',
  'float',
  'vec2',
  'vec3',
  'vec4',
  'string',
  'name',
  'entity',
  'component',
];

const defaultValueForType = (type: FlowValueType): unknown => {
  switch (type) {
    case 'bool':
      return false;
    case 'int':
    case 'float':
      return 0;
    case 'vec2':
      return [0, 0];
    case 'vec3':
      return [0, 0, 0];
    case 'vec4':
      return [0, 0, 0, 0];
    case 'string':
    case 'name':
      return '';
    case 'entity':
    case 'component':
    case 'any':
      return null;
  }
};

const vectorValue = (value: unknown, size: number) => {
  if (Array.isArray(value) && value.length === size && value.every((entry) => typeof entry === 'number'))
    return value as number[];
  return Array.from({ length: size }, () => 0);
};

export function FlowEditor({ document }: { document: EditorDocument; context?: EditorSurfaceContext }) {
  const state = useFlowDocumentState(document);

  useEffect(() => () => disposeFlowDocument(document.id), [document.id]);

  if (!state.loaded)
    return (
      <div className="flow-editor-loading">
        <strong>{state.loading ? 'Loading Flow graph…' : 'Flow graph unavailable'}</strong>
        {state.message && <span>{state.message}</span>}
      </div>
    );

  const mutateGraph = (updater: (graph: FlowGraph) => void) => {
    const graph = structuredClone(state.graph);
    updater(graph);
    replaceFlowGraph(document, graph);
  };

  const mutateVariables = (updater: (variables: typeof state.graph.variables, graph: FlowGraph) => void) =>
    mutateGraph((graph) => updater(graph.variables, graph));

  return (
    <section className="flow-editor">
      <FlowGraphEditor document={document} graph={state.graph} />
      <aside className="flow-editor-inspector" aria-label="Flow graph details">
        <header>
          <div>
            <strong>Flow Graph</strong>
            <span>Gameplay logic</span>
          </div>
          <span className="flow-schema-badge">v{state.graph.version}</span>
        </header>

        <section className="flow-editor-summary">
          <dl>
            <div>
              <dt>Nodes</dt>
              <dd>{state.graph.nodes.length}</dd>
            </div>
            <div>
              <dt>Connections</dt>
              <dd>{state.graph.connections.length}</dd>
            </div>
            <div>
              <dt>Variables</dt>
              <dd>{state.graph.variables.length}</dd>
            </div>
          </dl>
          <p>
            Flow graphs compile to typed runtime bytecode; gameplay world operations execute through ARC's stable world
            API.
          </p>
        </section>

        <section className="flow-variable-panel">
          <div className="flow-variable-panel-heading">
            <div>
              <strong>Variables</strong>
              <span>Per-instance gameplay state</span>
            </div>
            <UiButton
              aria-label="Add Flow variable"
              disabled={document.readOnly}
              onClick={() =>
                mutateVariables((variables) => {
                  variables.push({
                    id: flowGraphId('variable'),
                    name: `Variable ${variables.length + 1}`,
                    type: 'float',
                    defaultValue: 0,
                    exposed: false,
                  });
                })
              }
              variant="ghost"
            >
              <Plus size={13} /> Add
            </UiButton>
          </div>

          <div className="flow-variable-list">
            {state.graph.variables.length === 0 && <p className="flow-variable-empty">No graph variables yet.</p>}
            {state.graph.variables.map((variable) => (
              <article className="flow-variable" key={variable.id}>
                <div className="flow-variable-name-row">
                  <input
                    aria-label={`Variable name ${variable.name}`}
                    disabled={document.readOnly}
                    onChange={(event) =>
                      mutateVariables((variables) => {
                        const target = variables.find((candidate) => candidate.id === variable.id);
                        if (target) target.name = event.target.value;
                      })
                    }
                    value={variable.name}
                  />
                  <UiButton
                    aria-label={`Delete variable ${variable.name}`}
                    disabled={document.readOnly}
                    onClick={() =>
                      mutateVariables((variables, graph) => {
                        const index = variables.findIndex((candidate) => candidate.id === variable.id);
                        if (index < 0) return;
                        variables.splice(index, 1);
                        const affected = new Set(
                          graph.nodes
                            .filter((node) => node.values.variableId === variable.id)
                            .map((node) => node.id),
                        );
                        for (const node of graph.nodes) {
                          if (!affected.has(node.id)) continue;
                          node.values.variableId = '';
                          node.values.variableType = 'float';
                        }
                        graph.connections = graph.connections.filter(
                          (connection) =>
                            connection.kind === 'execution' ||
                            (!affected.has(connection.from.nodeId) && !affected.has(connection.to.nodeId)),
                        );
                      })
                    }
                    variant="ghost"
                  >
                    <Trash2 size={13} />
                  </UiButton>
                </div>
                <div className="flow-variable-fields">
                  <label>
                    Type
                    <select
                      disabled={document.readOnly}
                      onChange={(event) =>
                        mutateVariables((variables, graph) => {
                          const target = variables.find((candidate) => candidate.id === variable.id);
                          if (!target) return;
                          target.type = event.target.value as FlowValueType;
                          target.defaultValue = defaultValueForType(target.type);
                          const affected = new Set(
                            graph.nodes
                              .filter((node) => node.values.variableId === variable.id)
                              .map((node) => node.id),
                          );
                          for (const node of graph.nodes)
                            if (affected.has(node.id)) node.values.variableType = target.type;
                          graph.connections = graph.connections.filter(
                            (connection) =>
                              connection.kind === 'execution' ||
                              (!affected.has(connection.from.nodeId) && !affected.has(connection.to.nodeId)),
                          );
                        })
                      }
                      value={variable.type}
                    >
                      {variableTypes.map((type) => (
                        <option key={type} value={type}>
                          {type}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="flow-variable-exposed">
                    <input
                      checked={variable.exposed}
                      disabled={document.readOnly}
                      onChange={(event) =>
                        mutateVariables((variables) => {
                          const target = variables.find((candidate) => candidate.id === variable.id);
                          if (target) target.exposed = event.target.checked;
                        })
                      }
                      type="checkbox"
                    />
                    Expose
                  </label>
                </div>

                {variable.type === 'bool' && (
                  <label className="flow-variable-default">
                    Default
                    <input
                      checked={variable.defaultValue === true}
                      disabled={document.readOnly}
                      onChange={(event) =>
                        mutateVariables((variables) => {
                          const target = variables.find((candidate) => candidate.id === variable.id);
                          if (target) target.defaultValue = event.target.checked;
                        })
                      }
                      type="checkbox"
                    />
                  </label>
                )}

                {(variable.type === 'int' || variable.type === 'float') && (
                  <label className="flow-variable-default">
                    Default
                    <input
                      disabled={document.readOnly}
                      onChange={(event) => {
                        const value = Number(event.target.value);
                        mutateVariables((variables) => {
                          const target = variables.find((candidate) => candidate.id === variable.id);
                          if (target)
                            target.defaultValue = Number.isFinite(value)
                              ? variable.type === 'int'
                                ? Math.trunc(value)
                                : value
                              : 0;
                        });
                      }}
                      step={variable.type === 'int' ? 1 : 'any'}
                      type="number"
                      value={typeof variable.defaultValue === 'number' ? variable.defaultValue : 0}
                    />
                  </label>
                )}

                {(variable.type === 'string' || variable.type === 'name') && (
                  <label className="flow-variable-default">
                    Default
                    <input
                      disabled={document.readOnly}
                      onChange={(event) =>
                        mutateVariables((variables) => {
                          const target = variables.find((candidate) => candidate.id === variable.id);
                          if (target) target.defaultValue = event.target.value;
                        })
                      }
                      value={typeof variable.defaultValue === 'string' ? variable.defaultValue : ''}
                    />
                  </label>
                )}

                {(variable.type === 'vec2' || variable.type === 'vec3' || variable.type === 'vec4') &&
                  (() => {
                    const size = variable.type === 'vec2' ? 2 : variable.type === 'vec3' ? 3 : 4;
                    const values = vectorValue(variable.defaultValue, size);
                    return (
                      <label className="flow-variable-default">
                        Default
                        <span style={{ display: 'grid', gap: 4, gridTemplateColumns: `repeat(${size}, 1fr)` }}>
                          {values.map((entry, component) => (
                            <input
                              aria-label={`${variable.name} default component ${component + 1}`}
                              disabled={document.readOnly}
                              key={component}
                              onChange={(event) => {
                                const next = [...values];
                                const parsed = Number(event.target.value);
                                next[component] = Number.isFinite(parsed) ? parsed : 0;
                                mutateVariables((variables) => {
                                  const target = variables.find((candidate) => candidate.id === variable.id);
                                  if (target) target.defaultValue = next;
                                });
                              }}
                              step="any"
                              type="number"
                              value={entry}
                            />
                          ))}
                        </span>
                      </label>
                    );
                  })()}
              </article>
            ))}
          </div>
        </section>

        {state.message && <footer className="flow-editor-message">{state.message}</footer>}
      </aside>
    </section>
  );
}