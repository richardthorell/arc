import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Plus, Search, Trash2 } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import {
  GraphPin,
  GraphSelectionBox,
  GraphViewportLayer,
  GraphWireLayer,
  clampGraphZoom,
  clientToGraphPoint,
  graphConnectionPath,
  graphPinKey,
  graphPinPositionsChanged,
  graphSelectionBounds,
  graphSelectionScreenRect,
  measureGraphPinPositions,
  type GraphPoint,
  type GraphSelection,
} from '../graph';
import { UiButton, UiContextMenu, UiContextMenuItem, UiNodeCard, UiTextInput } from '../ui';
import { redoFlowGraph, replaceFlowGraph, undoFlowGraph } from './flowDocumentState';
import { flowGraphDomain } from './flowGraphDomain';
import {
  cloneFlowGraph,
  createFlowNode,
  flowGraphId,
  type FlowGraph,
  type FlowGraphNode,
  type FlowGraphConnection,
  type FlowNodeType,
  type FlowPinType,
} from './flowGraphTypes';

const nodeWidth = 238;
const headerHeight = 34;
const pinRowHeight = 25;
const nodePaddingTop = 9;

type PendingConnection = {
  nodeId: string;
  pin: string;
};

type FlowClipboard = {
  nodes: FlowGraphNode[];
  connections: FlowGraphConnection[];
};

let flowClipboard: FlowClipboard | null = null;

const pinTypeLabel = (type: FlowPinType) => (type.kind === 'execution' ? 'Execution' : type.valueType);
const pinTypeClass = (type: FlowPinType) => (type.kind === 'execution' ? 'execution' : `value value-${type.valueType}`);

const pinY = (node: FlowGraphNode, pin: string, output: boolean) => {
  const definition = flowGraphDomain.getNodeDefinition(node);
  const pins = output ? definition.outputs : definition.inputs;
  const index = Math.max(
    0,
    pins.findIndex((candidate) => candidate.id === pin),
  );
  return node.position[1] + headerHeight + nodePaddingTop + pinRowHeight * index + pinRowHeight / 2;
};

const fallbackPinPosition = (node: FlowGraphNode, pin: string, output: boolean): GraphPoint => [
  node.position[0] + (output ? nodeWidth : 0),
  pinY(node, pin, output),
];

const nodeHeight = (node: FlowGraphNode) => {
  const definition = flowGraphDomain.getNodeDefinition(node);
  return (
    headerHeight + nodePaddingTop + Math.max(definition.inputs.length, definition.outputs.length, 1) * pinRowHeight + 48
  );
};

export function FlowGraphEditor({ document, graph }: { document: EditorDocument; graph: FlowGraph }) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(() => new Set());
  const [pendingConnection, setPendingConnection] = useState<PendingConnection | null>(null);
  const [pointerGraph, setPointerGraph] = useState<GraphPoint>([0, 0]);
  const [pinPositions, setPinPositions] = useState<Map<string, GraphPoint>>(() => new Map());
  const [drag, setDrag] = useState<{ start: GraphPoint; nodes: Map<string, GraphPoint> } | null>(null);
  const [pan, setPan] = useState<{ start: GraphPoint; viewport: GraphPoint } | null>(null);
  const [box, setBox] = useState<GraphSelection | null>(null);
  const [addMenu, setAddMenu] = useState<{ screen: GraphPoint; graph: GraphPoint } | null>(null);
  const [nodeSearch, setNodeSearch] = useState('');
  const viewport = useMemo(() => graph.viewport ?? { x: 40, y: 40, zoom: 1 }, [graph.viewport]);

  useEffect(() => {
    setSelectedNodes((current) => new Set([...current].filter((id) => graph.nodes.some((node) => node.id === id))));
  }, [graph.nodes]);

  const mutate = useCallback(
    (updater: (draft: FlowGraph) => void, recordHistory = true) => {
      if (document.readOnly) return;
      const next = cloneFlowGraph(graph);
      updater(next);
      replaceFlowGraph(document, next, { recordHistory });
    },
    [document, graph],
  );

  const graphPoint = useCallback(
    (clientX: number, clientY: number): GraphPoint => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return [0, 0];
      return clientToGraphPoint(rect, viewport, clientX, clientY);
    },
    [viewport],
  );

  const measurePins = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const next = measureGraphPinPositions(canvas, viewport);
    setPinPositions((current) => (graphPinPositionsChanged(current, next) ? next : current));
  }, [viewport]);

  useLayoutEffect(() => {
    measurePins();
    const frame = window.requestAnimationFrame(measurePins);
    const nodes = canvasRef.current?.querySelectorAll<HTMLElement>('.flow-graph-node') ?? [];
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measurePins);
    for (const node of nodes) observer?.observe(node);
    window.addEventListener('resize', measurePins);
    return () => {
      window.cancelAnimationFrame(frame);
      observer?.disconnect();
      window.removeEventListener('resize', measurePins);
    };
  }, [graph.nodes, measurePins]);

  const updateViewport = useCallback(
    (patch: Partial<typeof viewport>, recordHistory = false) =>
      mutate((next) => {
        next.viewport = { ...viewport, ...patch };
      }, recordHistory),
    [mutate, viewport],
  );

  useEffect(() => {
    if (!drag && !pan && !box) return;
    const move = (event: PointerEvent) => {
      const point = graphPoint(event.clientX, event.clientY);
      setPointerGraph(point);
      if (drag) {
        const deltaX = point[0] - drag.start[0];
        const deltaY = point[1] - drag.start[1];
        mutate((next) => {
          for (const node of next.nodes) {
            const origin = drag.nodes.get(node.id);
            if (origin) node.position = [origin[0] + deltaX, origin[1] + deltaY];
          }
        }, false);
      } else if (pan) {
        updateViewport(
          {
            x: pan.viewport[0] + event.clientX - pan.start[0],
            y: pan.viewport[1] + event.clientY - pan.start[1],
          },
          false,
        );
      } else if (box) {
        setBox({ ...box, current: point });
      }
    };
    const up = () => {
      if (drag) replaceFlowGraph(document, graph, { recordHistory: true });
      if (box) {
        const bounds = graphSelectionBounds(box);
        setSelectedNodes(
          new Set(
            graph.nodes
              .filter(
                (node) =>
                  node.position[0] + nodeWidth >= bounds.left &&
                  node.position[0] <= bounds.right &&
                  node.position[1] + nodeHeight(node) >= bounds.top &&
                  node.position[1] <= bounds.bottom,
              )
              .map((node) => node.id),
          ),
        );
      }
      setDrag(null);
      setPan(null);
      setBox(null);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up, { once: true });
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
  }, [box, document, drag, graph, graphPoint, mutate, pan, updateViewport]);

  const deleteSelected = useCallback(() => {
    if (document.readOnly || selectedNodes.size === 0) return;
    mutate((next) => {
      const removable = new Set(
        [...selectedNodes].filter((id) => {
          const node = next.nodes.find((candidate) => candidate.id === id);
          return node ? flowGraphDomain.canDeleteNode(node) : false;
        }),
      );
      next.nodes = next.nodes.filter((node) => !removable.has(node.id));
      next.connections = next.connections.filter(
        (connection) => !removable.has(connection.from.nodeId) && !removable.has(connection.to.nodeId),
      );
    });
    setSelectedNodes(new Set());
  }, [document.readOnly, mutate, selectedNodes]);

  const copySelected = useCallback(() => {
    const nodes = graph.nodes.filter((node) => selectedNodes.has(node.id)).map((node) => ({ ...node }));
    const nodeIds = new Set(nodes.map((node) => node.id));
    flowClipboard = {
      nodes: cloneFlowGraph({ ...graph, nodes, connections: [] }).nodes,
      connections: graph.connections.filter(
        (connection) => nodeIds.has(connection.from.nodeId) && nodeIds.has(connection.to.nodeId),
      ),
    };
  }, [graph, selectedNodes]);

  const pasteClipboard = useCallback(() => {
    if (document.readOnly || !flowClipboard?.nodes.length) return;
    const idMap = new Map<string, string>();
    const nodes = flowClipboard.nodes.map((source) => {
      const id = flowGraphId(source.type);
      idMap.set(source.id, id);
      return {
        ...source,
        id,
        position: [source.position[0] + 36, source.position[1] + 36] as GraphPoint,
      };
    });
    const connections = flowClipboard.connections.map((connection) => ({
      ...connection,
      id: flowGraphId('connection'),
      from: { ...connection.from, nodeId: idMap.get(connection.from.nodeId) ?? connection.from.nodeId },
      to: { ...connection.to, nodeId: idMap.get(connection.to.nodeId) ?? connection.to.nodeId },
    }));
    mutate((next) => {
      next.nodes.push(...nodes);
      next.connections.push(...connections);
    });
    setSelectedNodes(new Set(nodes.map((node) => node.id)));
  }, [document.readOnly, mutate]);

  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement
      )
        return;
      const command = event.ctrlKey || event.metaKey;
      const key = event.key.toLocaleLowerCase();
      if ((event.key === 'Delete' || event.key === 'Backspace') && selectedNodes.size) {
        event.preventDefault();
        deleteSelected();
      } else if (command && key === 'c') {
        event.preventDefault();
        copySelected();
      } else if (command && key === 'v') {
        event.preventDefault();
        pasteClipboard();
      } else if (command && key === 'd') {
        event.preventDefault();
        copySelected();
        pasteClipboard();
      } else if (command && key === 'z') {
        event.preventDefault();
        if (event.shiftKey) redoFlowGraph(document);
        else undoFlowGraph(document);
      } else if (command && key === 'y') {
        event.preventDefault();
        redoFlowGraph(document);
      }
    };
    window.addEventListener('keydown', keyDown);
    return () => window.removeEventListener('keydown', keyDown);
  }, [copySelected, deleteSelected, document, pasteClipboard, selectedNodes.size]);

  const connectTo = (target: PendingConnection) => {
    if (!pendingConnection || document.readOnly) return;
    const fromNode = graph.nodes.find((node) => node.id === pendingConnection.nodeId);
    const toNode = graph.nodes.find((node) => node.id === target.nodeId);
    const fromPin = fromNode
      ? flowGraphDomain.getNodeDefinition(fromNode).outputs.find((pin) => pin.id === pendingConnection.pin)
      : undefined;
    const toPin = toNode
      ? flowGraphDomain.getNodeDefinition(toNode).inputs.find((pin) => pin.id === target.pin)
      : undefined;
    const validation =
      fromNode && fromPin && toNode && toPin
        ? flowGraphDomain.canConnect(
            { node: fromNode, pin: fromPin, direction: 'output' },
            { node: toNode, pin: toPin, direction: 'input' },
          )
        : { allowed: false as const };
    if (!validation.allowed || !fromPin) {
      setPendingConnection(null);
      return;
    }

    mutate((next) => {
      next.connections = next.connections.filter(
        (connection) => !(connection.to.nodeId === target.nodeId && connection.to.pin === target.pin),
      );
      next.connections.push({
        id: flowGraphId('connection'),
        kind: fromPin.type.kind,
        from: pendingConnection,
        to: target,
      });
    });
    setPendingConnection(null);
  };

  const addNode = (type: FlowNodeType) => {
    if (document.readOnly || !addMenu) return;
    const node = createFlowNode(type, addMenu.graph);
    mutate((next) => next.nodes.push(node));
    setSelectedNodes(new Set([node.id]));
    setAddMenu(null);
    setNodeSearch('');
  };

  const availableNodes = useMemo(() => {
    const query = nodeSearch.trim().toLocaleLowerCase();
    return flowGraphDomain
      .getNodeDefinitions()
      .filter(
        (definition) =>
          !query ||
          `${definition.title} ${definition.category} ${definition.subcategory ?? ''}`
            .toLocaleLowerCase()
            .includes(query),
      );
  }, [nodeSearch]);

  const pinPosition = (node: FlowGraphNode, pin: string, output: boolean) =>
    pinPositions.get(graphPinKey(node.id, pin, output)) ?? fallbackPinPosition(node, pin, output);

  const wires = graph.connections.flatMap((connection) => {
    const fromNode = graph.nodes.find((node) => node.id === connection.from.nodeId);
    const toNode = graph.nodes.find((node) => node.id === connection.to.nodeId);
    if (!fromNode || !toNode) return [];
    return [
      {
        id: connection.id,
        path: graphConnectionPath(
          pinPosition(fromNode, connection.from.pin, true),
          pinPosition(toNode, connection.to.pin, false),
        ),
      },
    ];
  });

  const pendingPath = (() => {
    if (!pendingConnection) return null;
    const node = graph.nodes.find((candidate) => candidate.id === pendingConnection.nodeId);
    if (!node) return null;
    return graphConnectionPath(pinPosition(node, pendingConnection.pin, true), pointerGraph);
  })();

  return (
    <div
      aria-label="Flow graph"
      className={`flow-graph-canvas ${document.readOnly ? 'read-only' : ''}`}
      onContextMenu={(event) => {
        event.preventDefault();
        if (document.readOnly) return;
        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect) return;
        setNodeSearch('');
        setAddMenu({
          screen: [event.clientX - rect.left, event.clientY - rect.top],
          graph: graphPoint(event.clientX, event.clientY),
        });
      }}
      onPointerDown={(event) => {
        if (event.target !== event.currentTarget) return;
        const point = graphPoint(event.clientX, event.clientY);
        setPointerGraph(point);
        setAddMenu(null);
        if (event.button === 1 || event.altKey) {
          event.preventDefault();
          setPan({ start: [event.clientX, event.clientY], viewport: [viewport.x, viewport.y] });
          return;
        }
        if (event.button === 0) {
          setSelectedNodes(new Set());
          setPendingConnection(null);
          setBox({ start: point, current: point });
        }
      }}
      onPointerMove={(event) => setPointerGraph(graphPoint(event.clientX, event.clientY))}
      onWheel={(event) => {
        event.preventDefault();
        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect) return;
        const before = graphPoint(event.clientX, event.clientY);
        const zoom = clampGraphZoom(viewport.zoom * (event.deltaY > 0 ? 0.9 : 1.1));
        updateViewport(
          {
            x: event.clientX - rect.left - before[0] * zoom,
            y: event.clientY - rect.top - before[1] * zoom,
            zoom,
          },
          false,
        );
      }}
      ref={canvasRef}
      role="application"
      tabIndex={0}
    >
      <div className="flow-graph-canvas-actions">
        <UiButton
          disabled={document.readOnly}
          onClick={() => {
            const rect = canvasRef.current?.getBoundingClientRect();
            if (!rect) return;
            const screen: GraphPoint = [24, 48];
            setNodeSearch('');
            setAddMenu({ screen, graph: graphPoint(rect.left + screen[0], rect.top + screen[1]) });
          }}
          variant="ghost"
        >
          <Plus size={13} /> Add Node
        </UiButton>
        <UiButton disabled={document.readOnly || !selectedNodes.size} onClick={deleteSelected} variant="ghost">
          <Trash2 size={13} /> Delete
        </UiButton>
        <span>{Math.round(viewport.zoom * 100)}%</span>
      </div>

      <GraphViewportLayer className="flow-graph-transform" viewport={viewport}>
        <GraphWireLayer className="flow-graph-wires" pendingPath={pendingPath} wires={wires} />
        {graph.nodes.map((node) => {
          const definition = flowGraphDomain.getNodeDefinition(node);
          const selected = selectedNodes.has(node.id);
          return (
            <UiNodeCard
              badge={definition.category === 'Events' || definition.category === 'Input' ? 'E' : undefined}
              badgeTitle={
                definition.category === 'Events' || definition.category === 'Input' ? 'Event entry point' : undefined
              }
              className={`flow-graph-node flow-graph-node-${node.type}`}
              data-node-id={node.id}
              heading={definition.title}
              key={node.id}
              onHeaderPointerDown={(event) => {
                if (document.readOnly || event.button !== 0) return;
                event.preventDefault();
                event.stopPropagation();
                const selection = selected ? selectedNodes : new Set([node.id]);
                if (!selected) setSelectedNodes(selection);
                const origins = new Map<string, GraphPoint>();
                for (const candidate of graph.nodes)
                  if (selection.has(candidate.id)) origins.set(candidate.id, [...candidate.position]);
                setDrag({ start: graphPoint(event.clientX, event.clientY), nodes: origins });
              }}
              onPointerDown={(event) => {
                if (event.button !== 0) return;
                event.stopPropagation();
                if (!event.ctrlKey && !event.metaKey && !selected) setSelectedNodes(new Set([node.id]));
                else if (event.ctrlKey || event.metaKey) {
                  setSelectedNodes((current) => {
                    const next = new Set(current);
                    if (next.has(node.id)) next.delete(node.id);
                    else next.add(node.id);
                    return next;
                  });
                }
              }}
              selected={selected}
              style={{ left: node.position[0], top: node.position[1], width: nodeWidth }}
              tone={definition.category === 'Events' || definition.category === 'Input' ? 'accent' : 'default'}
            >
              <div className="flow-node-pins">
                <div className="flow-node-inputs">
                  {definition.inputs.map((pin) => (
                    <GraphPin
                      className={`flow-pin ${pinTypeClass(pin.type)}`}
                      connected={graph.connections.some(
                        (connection) => connection.to.nodeId === node.id && connection.to.pin === pin.id,
                      )}
                      direction="input"
                      disabled={document.readOnly}
                      key={pin.id}
                      label={pin.label}
                      onPointerDown={(event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        if (pendingConnection) connectTo({ nodeId: node.id, pin: pin.id });
                      }}
                      pinKey={graphPinKey(node.id, pin.id, false)}
                      title={`${pin.label} · ${pinTypeLabel(pin.type)}`}
                    />
                  ))}
                </div>
                <div className="flow-node-outputs">
                  {definition.outputs.map((pin) => (
                    <GraphPin
                      className={`flow-pin ${pinTypeClass(pin.type)}`}
                      connected={graph.connections.some(
                        (connection) => connection.from.nodeId === node.id && connection.from.pin === pin.id,
                      )}
                      direction="output"
                      disabled={document.readOnly}
                      key={pin.id}
                      label={pin.label}
                      onPointerDown={(event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        setPendingConnection({ nodeId: node.id, pin: pin.id });
                        setPointerGraph(graphPoint(event.clientX, event.clientY));
                      }}
                      pinKey={graphPinKey(node.id, pin.id, true)}
                      title={`${pin.label} · ${pinTypeLabel(pin.type)}`}
                    />
                  ))}
                </div>
              </div>

              {node.type === 'inputAction' && (
                <label className="flow-node-inline-value">
                  Action
                  <input
                    aria-label="Input action"
                    disabled={document.readOnly}
                    onChange={(event) =>
                      mutate((next) => {
                        const target = next.nodes.find((candidate) => candidate.id === node.id);
                        if (target) target.values.action = event.target.value;
                      })
                    }
                    value={typeof node.values.action === 'string' ? node.values.action : ''}
                  />
                </label>
              )}
            </UiNodeCard>
          );
        })}
      </GraphViewportLayer>

      {box && <GraphSelectionBox className="flow-graph-box-selection" rect={graphSelectionScreenRect(box, viewport)} />}

      {addMenu && (
        <UiContextMenu
          aria-label="Add Flow node"
          className="flow-node-menu"
          maxHeight={420}
          width={300}
          x={addMenu.screen[0]}
          y={addMenu.screen[1]}
        >
          <div className="flow-node-menu-search">
            <Search size={13} />
            <UiTextInput
              aria-label="Search Flow nodes"
              autoFocus
              onChange={(event) => setNodeSearch(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Escape') setAddMenu(null);
                else if (event.key === 'Enter' && availableNodes[0]) addNode(availableNodes[0].type);
              }}
              placeholder="Search events and nodes"
              value={nodeSearch}
            />
          </div>
          <div className="flow-node-menu-items">
            {availableNodes.map((definition) => (
              <UiContextMenuItem
                key={definition.type}
                onClick={() => addNode(definition.type)}
                trailing={<small>{`${definition.category} / ${definition.subcategory ?? 'General'}`}</small>}
              >
                <strong>{definition.title}</strong>
              </UiContextMenuItem>
            ))}
          </div>
        </UiContextMenu>
      )}
    </div>
  );
}
