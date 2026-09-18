import { useEffect } from 'react';
import { Plus, Trash2 } from 'lucide-react';

import type { EditorDocument, EditorSurfaceContext } from '../editors/editorTypes';
import { UiButton } from '../ui';
import { FlowGraphEditor } from './FlowGraphEditor';
import { disposeFlowDocument, replaceFlowGraph, useFlowDocumentState } from './flowDocumentState';
import { flowGraphId, type FlowGraph, type FlowInterfaceValueDefinition, type FlowValueType } from './flowGraphTypes';
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

  const mutateInterface = (
    kind: 'inputs' | 'outputs',
    updater: (values: FlowInterfaceValueDefinition[], graph: FlowGraph) => void,
  ) =>
    mutateGraph((graph) => {
      const values = graph[kind] ?? [];
      graph[kind] = values;
      updater(values, graph);
    });

  const mutateEvents = (updater: (events: NonNullable<FlowGraph['events']>, graph: FlowGraph) => void) =>
    mutateGraph((graph) => {
      const events = graph.events ?? [];
      graph.events = events;
      updater(events, graph);
    });

  const renderInterfacePanel = (kind: 'inputs' | 'outputs', title: string) => {
    const values = state.graph[kind] ?? [];
    const nodeType = kind === 'inputs' ? 'graphInput' : 'graphOutput';
    return (
      <section className="flow-variable-panel">
        <div className="flow-variable-panel-heading">
          <div>
            <strong>{title}</strong>
            <span>{kind === 'inputs' ? 'Values supplied to this graph' : 'Values produced by this graph'}</span>
          </div>
          <UiButton
            aria-label={`Add graph ${kind === 'inputs' ? 'input' : 'output'}`}
            disabled={document.readOnly}
            onClick={() =>
              mutateInterface(kind, (items) => {
                items.push({
                  id: flowGraphId(kind === 'inputs' ? 'input' : 'output'),
                  name: `${kind === 'inputs' ? 'Input' : 'Output'} ${items.length + 1}`,
                  type: 'float',
                  defaultValue: 0,
                });
              })
            }
            variant="ghost"
          >
            <Plus size={13} /> Add
          </UiButton>
        </div>
        <div className="flow-variable-list">
          {values.length === 0 && <p className="flow-variable-empty">No {kind} yet.</p>}
          {values.map((item) => (
            <article className="flow-variable" key={item.id}>
              <div className="flow-variable-name-row">
                <input
                  aria-label={`${title} name ${item.name}`}
                  disabled={document.readOnly}
                  onChange={(event) =>
                    mutateInterface(kind, (items) => {
                      const target = items.find((candidate) => candidate.id === item.id);
                      if (target) target.name = event.target.value;
                    })
                  }
                  value={item.name}
                />
                <UiButton
                  aria-label={`Delete ${title.toLowerCase()} ${item.name}`}
                  disabled={document.readOnly}
                  onClick={() =>
                    mutateInterface(kind, (items, graph) => {
                      const index = items.findIndex((candidate) => candidate.id === item.id);
                      if (index < 0) return;
                      items.splice(index, 1);
                      const affected = new Set(
                        graph.nodes
                          .filter((node) => node.type === nodeType && node.values.interfaceId === item.id)
                          .map((node) => node.id),
                      );
                      for (const node of graph.nodes) {
                        if (!affected.has(node.id)) continue;
                        node.values.interfaceId = '';
                        node.values.interfaceType = 'float';
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
                      mutateInterface(kind, (items, graph) => {
                        const target = items.find((candidate) => candidate.id === item.id);
                        if (!target) return;
                        target.type = event.target.value as FlowValueType;
                        target.defaultValue = defaultValueForType(target.type);
                        const affected = new Set(
                          graph.nodes
                            .filter((node) => node.type === nodeType && node.values.interfaceId === item.id)
                            .map((node) => node.id),
                        );
                        for (const node of graph.nodes)
                          if (affected.has(node.id)) node.values.interfaceType = target.type;
                        graph.connections = graph.connections.filter(
                          (connection) =>
                            connection.kind === 'execution' ||
                            (!affected.has(connection.from.nodeId) && !affected.has(connection.to.nodeId)),
                        );
                      })
                    }
                    value={item.type}
                  >
                    {variableTypes.map((type) => (
                      <option key={type} value={type}>
                        {type}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </article>
          ))}
        </div>
      </section>
    );
  };

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
            <div>
              <dt>Interface</dt>
              <dd>{(state.graph.inputs?.length ?? 0) + (state.graph.outputs?.length ?? 0)}</dd>
            </div>
            <div>
              <dt>Events</dt>
              <dd>{state.graph.events?.length ?? 0}</dd>
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
              <strong>Custom Events</strong>
              <span>Local entry points callable by name</span>
            </div>
            <UiButton
              aria-label="Add custom event"
              disabled={document.readOnly}
              onClick={() =>
                mutateEvents((events) => {
                  events.push({ id: flowGraphId('event'), name: `Event ${events.length + 1}` });
                })
              }
              variant="ghost"
            >
              <Plus size={13} /> Add
            </UiButton>
          </div>
          <div className="flow-variable-list">
            {(state.graph.events?.length ?? 0) === 0 && <p className="flow-variable-empty">No custom events yet.</p>}
            {(state.graph.events ?? []).map((event) => (
              <article className="flow-variable" key={event.id}>
                <div className="flow-variable-name-row">
                  <input
                    aria-label={`Custom event name ${event.name}`}
                    disabled={document.readOnly}
                    onChange={(change) =>
                      mutateEvents((events) => {
                        const target = events.find((candidate) => candidate.id === event.id);
                        if (target) target.name = change.target.value;
                      })
                    }
                    value={event.name}
                  />
                  <UiButton
                    aria-label={`Delete custom event ${event.name}`}
                    disabled={document.readOnly}
                    onClick={() =>
                      mutateEvents((events, graph) => {
                        const index = events.findIndex((candidate) => candidate.id === event.id);
                        if (index < 0) return;
                        events.splice(index, 1);
                        for (const node of graph.nodes)
                          if (
                            (node.type === 'customEvent' || node.type === 'callCustomEvent') &&
                            node.values.eventId === event.id
                          )
                            node.values.eventId = '';
                      })
                    }
                    variant="ghost"
                  >
                    <Trash2 size={13} />
                  </UiButton>
                </div>
              </article>
            ))}
          </div>
        </section>

        {renderInterfacePanel('inputs', 'Graph Inputs')}
        {renderInterfacePanel('outputs', 'Graph Outputs')}

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
                          graph.nodes.filter((node) => node.values.variableId === variable.id).map((node) => node.id),
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
                            graph.nodes.filter((node) => node.values.variableId === variable.id).map((node) => node.id),
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
