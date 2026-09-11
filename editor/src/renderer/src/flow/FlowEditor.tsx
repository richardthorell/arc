import { useEffect } from 'react';
import { Plus, Trash2 } from 'lucide-react';

import type { EditorDocument, EditorSurfaceContext } from '../editors/editorTypes';
import { UiButton } from '../ui';
import { FlowGraphEditor } from './FlowGraphEditor';
import { disposeFlowDocument, replaceFlowGraph, useFlowDocumentState } from './flowDocumentState';
import { flowGraphId, type FlowValueType } from './flowGraphTypes';
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

  const mutateVariables = (updater: (variables: typeof state.graph.variables) => void) => {
    const graph = structuredClone(state.graph);
    updater(graph.variables);
    replaceFlowGraph(document, graph);
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
          </dl>
          <p>
            Flow graphs are authoring data. Compilation and runtime execution are introduced in the next milestones.
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
                      mutateVariables((variables) => {
                        const index = variables.findIndex((candidate) => candidate.id === variable.id);
                        if (index >= 0) variables.splice(index, 1);
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
                        mutateVariables((variables) => {
                          const target = variables.find((candidate) => candidate.id === variable.id);
                          if (!target) return;
                          target.type = event.target.value as FlowValueType;
                          target.defaultValue = defaultValueForType(target.type);
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
              </article>
            ))}
          </div>
        </section>

        {state.message && <footer className="flow-editor-message">{state.message}</footer>}
      </aside>
    </section>
  );
}
