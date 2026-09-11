import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { ChevronRight, Copy, Plus, Search, Trash2 } from 'lucide-react';

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
import { UiButton, UiColorControl, UiContextMenu, UiContextMenuItem, UiNodeCard, UiTextInput } from '../ui';
import { materialGraphDomain } from './materialGraphDomain';
import {
  cloneMaterialGraph,
  createMaterialNode,
  materialGraphId,
  materialNodeCategoryOrder,
  materialNodeSubcategoryOrder,
  type MaterialGraph,
  type MaterialGraphConnection,
  type MaterialGraphNode,
  type MaterialGraphNodeType,
  type MaterialGraphPinRef,
  type MaterialNodeCategory,
  type MaterialNodeSubcategory,
} from './materialGraphTypes';
import { redoMaterialGraph, replaceMaterialGraph, undoMaterialGraph } from './materialDocumentState';
import { MaterialTextureSampleEditor } from './MaterialTextureSampleEditor';

const defaultNodeWidth = 214;
const headerHeight = 34;
const pinRowHeight = 25;
const nodePaddingTop = 9;

type AddMenuCategory = Exclude<MaterialNodeCategory, 'Output'>;

export const materialNodeWidth = (type: MaterialGraphNodeType) => {
  switch (type) {
    case 'vector2':
      return 232;
    case 'vector3':
      return 252;
    case 'vector4':
      return 286;
    case 'colorRgb':
    case 'colorRgba':
      return 300;
    case 'textureSample':
      return 286;
    case 'output':
      return 236;
    case 'normalMap':
    case 'clamp':
      return 232;
    default:
      return defaultNodeWidth;
  }
};

const editableValueNode = (node: MaterialGraphNode) =>
  node.type === 'constant' ||
  node.type === 'vector2' ||
  node.type === 'vector3' ||
  node.type === 'vector4' ||
  node.type === 'colorRgb' ||
  node.type === 'colorRgba';

const pinY = (node: MaterialGraphNode, pin: string, output: boolean) => {
  const definition = materialGraphDomain.getNodeDefinition(node);
  const pins = output ? definition.outputs : definition.inputs;
  const index = Math.max(
    0,
    pins.findIndex((candidate) => candidate.id === pin),
  );
  return node.position[1] + headerHeight + nodePaddingTop + pinRowHeight * index + pinRowHeight / 2;
};

const fallbackPinPosition = (node: MaterialGraphNode, pin: string, output: boolean): GraphPoint => [
  node.position[0] + (output ? materialNodeWidth(node.type) : 0),
  pinY(node, pin, output),
];

type GraphClipboard = {
  nodes: MaterialGraphNode[];
  connections: MaterialGraphConnection[];
};

let graphClipboard: GraphClipboard | null = null;

const nextNodeValue = (node: MaterialGraphNode, value: unknown): MaterialGraphNode => ({
  ...node,
  values: { ...node.values, value },
});

const colorChannel = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? value : 0);
const colorTuple = (value: unknown): [number, number, number, number] => {
  const components = Array.isArray(value) ? value : [];
  return [
    colorChannel(components[0] ?? 1),
    colorChannel(components[1] ?? 1),
    colorChannel(components[2] ?? 1),
    colorChannel(components[3] ?? 1),
  ];
};

function MaterialNodeValueEditor({
  node,
  readOnly,
  onChange,
}: {
  node: MaterialGraphNode;
  readOnly: boolean;
  onChange: (node: MaterialGraphNode) => void;
}) {
  if (node.type === 'constant')
    return (
      <label className="material-node-inline-value">
        Value
        <input
          disabled={readOnly}
          type="number"
          step="0.01"
          value={typeof node.values.value === 'number' ? node.values.value : 0}
          onChange={(event) => onChange(nextNodeValue(node, Number(event.target.value)))}
        />
      </label>
    );

  if (node.type === 'vector2' || node.type === 'vector3' || node.type === 'vector4') {
    const size = node.type === 'vector2' ? 2 : node.type === 'vector3' ? 3 : 4;
    const current = Array.isArray(node.values.value) ? node.values.value : [];
    return (
      <div className="material-node-vector-value">
        {Array.from({ length: size }, (_, index) => (
          <input
            aria-label={`${node.type} component ${index + 1}`}
            disabled={readOnly}
            key={index}
            type="number"
            step="0.01"
            value={typeof current[index] === 'number' ? current[index] : 0}
            onChange={(event) => {
              const next = Array.from({ length: size }, (_, component) =>
                typeof current[component] === 'number' ? current[component] : 0,
              );
              next[index] = Number(event.target.value);
              onChange(nextNodeValue(node, next));
            }}
          />
        ))}
      </div>
    );
  }

  if (node.type === 'colorRgb' || node.type === 'colorRgba') {
    const color = colorTuple(node.values.value);
    return (
      <UiColorControl
        allowAlpha={node.type === 'colorRgba'}
        label={node.type === 'colorRgba' ? 'Color' : 'Legacy RGB color'}
        onCommit={(next) => {
          if (readOnly) return;
          onChange(nextNodeValue(node, node.type === 'colorRgba' ? next : next.slice(0, 3)));
        }}
        value={color}
      />
    );
  }

  if (node.type === 'textureSample')
    return <MaterialTextureSampleEditor node={node} readOnly={readOnly} onChange={onChange} />;

  if (node.type === 'normalMap')
    return (
      <label className="material-node-inline-value">
        Strength
        <input
          disabled={readOnly}
          type="number"
          min="0"
          step="0.05"
          value={typeof node.values.strength === 'number' ? node.values.strength : 1}
          onChange={(event) => onChange({ ...node, values: { ...node.values, strength: Number(event.target.value) } })}
        />
      </label>
    );

  if (node.type === 'clamp')
    return (
      <div className="material-node-vector-value">
        {(['min', 'max'] as const).map((key) => (
          <input
            aria-label={`Clamp ${key}`}
            disabled={readOnly}
            key={key}
            type="number"
            step="0.05"
            value={typeof node.values[key] === 'number' ? node.values[key] : key === 'min' ? 0 : 1}
            onChange={(event) => onChange({ ...node, values: { ...node.values, [key]: Number(event.target.value) } })}
          />
        ))}
      </div>
    );

  return null;
}

export function MaterialGraphEditor({ document, graph }: { document: EditorDocument; graph: MaterialGraph }) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(() => new Set());
  const [pendingConnection, setPendingConnection] = useState<MaterialGraphPinRef | null>(null);
  const [pointerGraph, setPointerGraph] = useState<GraphPoint>([0, 0]);
  const [pinPositions, setPinPositions] = useState<Map<string, GraphPoint>>(() => new Map());
  const [drag, setDrag] = useState<{ start: GraphPoint; nodes: Map<string, GraphPoint> } | null>(null);
  const [pan, setPan] = useState<{ start: GraphPoint; viewport: GraphPoint } | null>(null);
  const [box, setBox] = useState<GraphSelection | null>(null);
  const [addMenu, setAddMenu] = useState<{ screen: GraphPoint; graph: GraphPoint } | null>(null);
  const [nodeSearch, setNodeSearch] = useState('');
  const [nodeMenuCategory, setNodeMenuCategory] = useState<AddMenuCategory | null>(null);
  const [nodeMenuSubcategory, setNodeMenuSubcategory] = useState<MaterialNodeSubcategory | null>(null);
  const viewport = useMemo(() => graph.viewport ?? { x: 40, y: 40, zoom: 1 }, [graph.viewport]);

  useEffect(() => {
    setSelectedNodes((current) => new Set([...current].filter((id) => graph.nodes.some((node) => node.id === id))));
  }, [graph.nodes]);

  const mutate = useCallback(
    (updater: (draft: MaterialGraph) => void, recordHistory = true) => {
      if (document.readOnly) return;
      const next = cloneMaterialGraph(graph);
      updater(next);
      replaceMaterialGraph(document, next, { recordHistory });
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

  const measurePinPositions = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const next = measureGraphPinPositions(canvas, viewport);
    setPinPositions((current) => (graphPinPositionsChanged(current, next) ? next : current));
  }, [viewport]);

  useLayoutEffect(() => {
    measurePinPositions();
    const frame = window.requestAnimationFrame(measurePinPositions);
    const nodes = canvasRef.current?.querySelectorAll<HTMLElement>('.material-graph-node') ?? [];
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measurePinPositions);
    for (const node of nodes) observer?.observe(node);
    window.addEventListener('resize', measurePinPositions);
    return () => {
      window.cancelAnimationFrame(frame);
      observer?.disconnect();
      window.removeEventListener('resize', measurePinPositions);
    };
  }, [graph.nodes, measurePinPositions]);

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
            x: pan.viewport[0] + (event.clientX - pan.start[0]),
            y: pan.viewport[1] + (event.clientY - pan.start[1]),
          },
          false,
        );
      } else if (box) {
        setBox({ ...box, current: point });
      }
    };
    const up = () => {
      if (drag) replaceMaterialGraph(document, graph, { recordHistory: true });
      if (box) {
        const bounds = graphSelectionBounds(box);
        setSelectedNodes(
          new Set(
            graph.nodes
              .filter(
                (node) =>
                  node.position[0] + materialNodeWidth(node.type) >= bounds.left &&
                  node.position[0] <= bounds.right &&
                  node.position[1] + 180 >= bounds.top &&
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

  const deleteSelected = () => {
    if (document.readOnly || selectedNodes.size === 0) return;
    mutate((next) => {
      const removable = new Set(
        [...selectedNodes].filter((id) => {
          const node = next.nodes.find((candidate) => candidate.id === id);
          return node ? materialGraphDomain.canDeleteNode(node) : false;
        }),
      );
      next.nodes = next.nodes.filter((node) => !removable.has(node.id));
      next.connections = next.connections.filter(
        (connection) => !removable.has(connection.from.nodeId) && !removable.has(connection.to.nodeId),
      );
    });
    setSelectedNodes(new Set());
  };

  const copySelected = () => {
    const copiedNodes = graph.nodes
      .filter((node) => selectedNodes.has(node.id) && materialGraphDomain.canDeleteNode(node))
      .map((node) => ({ ...node }));
    const ids = new Set(copiedNodes.map((node) => node.id));
    graphClipboard = {
      nodes: cloneMaterialGraph({ version: 1, nodes: copiedNodes, connections: [] }).nodes,
      connections: graph.connections.filter(
        (connection) => ids.has(connection.from.nodeId) && ids.has(connection.to.nodeId),
      ),
    };
  };

  const pasteClipboard = () => {
    if (document.readOnly || !graphClipboard?.nodes.length) return;
    const idMap = new Map<string, string>();
    const nodes = graphClipboard.nodes.map((source) => {
      const id = materialGraphId(source.type);
      idMap.set(source.id, id);
      return { ...source, id, position: [source.position[0] + 36, source.position[1] + 36] as [number, number] };
    });
    const connections = graphClipboard.connections.map((connection) => ({
      ...connection,
      id: materialGraphId('connection'),
      from: { ...connection.from, nodeId: idMap.get(connection.from.nodeId) ?? connection.from.nodeId },
      to: { ...connection.to, nodeId: idMap.get(connection.to.nodeId) ?? connection.to.nodeId },
    }));
    mutate((next) => {
      next.nodes.push(...nodes);
      next.connections.push(...connections);
    });
    setSelectedNodes(new Set(nodes.map((node) => node.id)));
  };

  const duplicateSelected = () => {
    copySelected();
    pasteClipboard();
  };

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
      if ((event.key === 'Delete' || event.key === 'Backspace') && selectedNodes.size) {
        event.preventDefault();
        deleteSelected();
      } else if (command && event.key.toLocaleLowerCase() === 'c') {
        event.preventDefault();
        copySelected();
      } else if (command && event.key.toLocaleLowerCase() === 'v') {
        event.preventDefault();
        pasteClipboard();
      } else if (command && event.key.toLocaleLowerCase() === 'd') {
        event.preventDefault();
        duplicateSelected();
      } else if (command && event.key.toLocaleLowerCase() === 'z') {
        event.preventDefault();
        if (event.shiftKey) redoMaterialGraph(document);
        else undoMaterialGraph(document);
      } else if (command && event.key.toLocaleLowerCase() === 'y') {
        event.preventDefault();
        redoMaterialGraph(document);
      }
    };
    window.addEventListener('keydown', keyDown);
    return () => window.removeEventListener('keydown', keyDown);
  });

  const connectTo = (target: MaterialGraphPinRef) => {
    if (!pendingConnection || document.readOnly) return;
    const fromNode = graph.nodes.find((node) => node.id === pendingConnection.nodeId);
    const toNode = graph.nodes.find((node) => node.id === target.nodeId);
    const fromPin = fromNode
      ? materialGraphDomain.getNodeDefinition(fromNode).outputs.find((pin) => pin.id === pendingConnection.pin)
      : undefined;
    const toPin = toNode
      ? materialGraphDomain.getNodeDefinition(toNode).inputs.find((pin) => pin.id === target.pin)
      : undefined;
    const allowed =
      fromNode &&
      toNode &&
      fromPin &&
      toPin &&
      materialGraphDomain.canConnect(
        { node: fromNode, pin: fromPin, direction: 'output' },
        { node: toNode, pin: toPin, direction: 'input' },
      ).allowed;
    if (!allowed) {
      setPendingConnection(null);
      return;
    }
    mutate((next) => {
      next.connections = next.connections.filter(
        (connection) => !(connection.to.nodeId === target.nodeId && connection.to.pin === target.pin),
      );
      next.connections.push({
        id: materialGraphId('connection'),
        from: pendingConnection,
        to: target,
      });
    });
    setPendingConnection(null);
  };

  const resetAddMenuPath = () => {
    setNodeMenuCategory(null);
    setNodeMenuSubcategory(null);
  };

  const addNode = (type: MaterialGraphNodeType) => {
    if (type === 'output' || document.readOnly || !addMenu) return;
    const node = createMaterialNode(type, addMenu.graph);
    mutate((next) => next.nodes.push(node));
    setSelectedNodes(new Set([node.id]));
    setAddMenu(null);
    setNodeSearch('');
    resetAddMenuPath();
  };

  const availableNodes = useMemo(() => {
    const query = nodeSearch.trim().toLocaleLowerCase();
    return materialGraphDomain
      .getNodeDefinitions()
      .filter((definition) => definition.type !== 'output' && definition.type !== 'colorRgb')
      .filter(
        (definition) =>
          !query ||
          `${definition.title} ${definition.category} ${definition.subcategory}`.toLocaleLowerCase().includes(query),
      );
  }, [nodeSearch]);

  const visibleCategories = materialNodeCategoryOrder.filter((category): category is AddMenuCategory =>
    availableNodes.some((definition) => definition.category === category),
  );
  const searchingNodes = nodeSearch.trim().length > 0;
  const pinPosition = (node: MaterialGraphNode, pin: string, output: boolean) =>
    pinPositions.get(graphPinKey(node.id, pin, output)) ?? fallbackPinPosition(node, pin, output);

  const wirePaths = graph.connections.flatMap((connection) => {
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
      aria-label="Material graph"
      className={`material-graph-canvas ${document.readOnly ? 'read-only' : ''}`}
      ref={canvasRef}
      role="application"
      tabIndex={0}
      onContextMenu={(event) => {
        event.preventDefault();
        if (document.readOnly) return;
        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect) return;
        resetAddMenuPath();
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
        const x = event.clientX - rect.left - before[0] * zoom;
        const y = event.clientY - rect.top - before[1] * zoom;
        updateViewport({ x, y, zoom }, false);
      }}
    >
      <div className="material-graph-canvas-actions">
        <UiButton
          disabled={document.readOnly}
          onClick={() => {
            const rect = canvasRef.current?.getBoundingClientRect();
            if (!rect) return;
            const screen: GraphPoint = [24, 48];
            resetAddMenuPath();
            setNodeSearch('');
            setAddMenu({ screen, graph: graphPoint(rect.left + screen[0], rect.top + screen[1]) });
          }}
          variant="ghost"
        >
          <Plus size={13} /> Add Node
        </UiButton>
        <UiButton disabled={!selectedNodes.size} onClick={copySelected} variant="ghost">
          <Copy size={13} /> Copy
        </UiButton>
        <UiButton disabled={document.readOnly || !selectedNodes.size} onClick={deleteSelected} variant="ghost">
          <Trash2 size={13} /> Delete
        </UiButton>
        <span>{Math.round(viewport.zoom * 100)}%</span>
      </div>

      <GraphViewportLayer className="material-graph-transform" viewport={viewport}>
        <GraphWireLayer className="material-graph-wires" pendingPath={pendingPath} wires={wirePaths} />
        {graph.nodes.map((node) => {
          const definition = materialGraphDomain.getNodeDefinition(node);
          const selected = selectedNodes.has(node.id);
          return (
            <UiNodeCard
              badge={node.parameter?.exposed ? 'P' : undefined}
              badgeTitle={node.parameter?.exposed ? `Parameter: ${node.parameter.name}` : undefined}
              className={`material-graph-node material-graph-node-${node.type}`}
              data-node-id={node.id}
              heading={definition.title}
              key={node.id}
              selected={selected}
              style={{ left: node.position[0], top: node.position[1], width: materialNodeWidth(node.type) }}
              tone={node.type === 'output' ? 'accent' : 'default'}
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
            >
              <div className="material-node-pins">
                <div className="material-node-inputs">
                  {definition.inputs.map((pin) => {
                    const connected = graph.connections.some(
                      (connection) => connection.to.nodeId === node.id && connection.to.pin === pin.id,
                    );
                    return (
                      <GraphPin
                        className="material-pin"
                        connected={connected}
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
                        title={`${pin.label} · ${pin.type}`}
                      />
                    );
                  })}
                </div>
                <div className="material-node-outputs">
                  {definition.outputs.map((pin) => (
                    <GraphPin
                      className="material-pin"
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
                      title={`${pin.label} · ${pin.type}`}
                    />
                  ))}
                </div>
              </div>

              <MaterialNodeValueEditor
                node={node}
                readOnly={document.readOnly}
                onChange={(updated) =>
                  mutate((next) => {
                    const index = next.nodes.findIndex((candidate) => candidate.id === updated.id);
                    if (index >= 0) next.nodes[index] = updated;
                  })
                }
              />

              {editableValueNode(node) && (
                <label className="material-node-parameter-toggle">
                  <input
                    checked={Boolean(node.parameter?.exposed)}
                    disabled={document.readOnly}
                    type="checkbox"
                    onChange={(event) =>
                      mutate((next) => {
                        const target = next.nodes.find((candidate) => candidate.id === node.id);
                        if (!target) return;
                        target.parameter = {
                          exposed: event.target.checked,
                          name: target.parameter?.name ?? definition.title,
                        };
                      })
                    }
                  />
                  <span>Parameter</span>
                  <input
                    aria-label="Parameter name"
                    disabled={document.readOnly || !node.parameter?.exposed}
                    value={node.parameter?.name ?? definition.title}
                    onChange={(event) =>
                      mutate((next) => {
                        const target = next.nodes.find((candidate) => candidate.id === node.id);
                        if (!target) return;
                        target.parameter = {
                          exposed: Boolean(target.parameter?.exposed),
                          name: event.target.value,
                        };
                      })
                    }
                  />
                </label>
              )}
            </UiNodeCard>
          );
        })}
      </GraphViewportLayer>

      {box && (
        <GraphSelectionBox className="material-graph-box-selection" rect={graphSelectionScreenRect(box, viewport)} />
      )}

      {addMenu && (
        <UiContextMenu
          aria-label="Add material node"
          className="material-node-menu"
          maxHeight={420}
          width={280}
          x={addMenu.screen[0]}
          y={addMenu.screen[1]}
        >
          <div className="material-node-menu-search">
            <Search size={13} />
            <UiTextInput
              aria-label="Search material nodes"
              autoFocus
              placeholder="Search nodes"
              value={nodeSearch}
              onChange={(event) => setNodeSearch(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Escape') setAddMenu(null);
                else if (event.key === 'Enter' && availableNodes[0]) addNode(availableNodes[0].type);
              }}
            />
          </div>
          <div className="material-node-menu-items">
            {searchingNodes
              ? availableNodes.map((definition) => (
                  <UiContextMenuItem
                    key={definition.type}
                    onClick={() => addNode(definition.type)}
                    trailing={<small>{`${definition.category} / ${definition.subcategory}`}</small>}
                  >
                    <strong>{definition.title}</strong>
                  </UiContextMenuItem>
                ))
              : visibleCategories.map((category) => {
                  const categoryActive = nodeMenuCategory === category;
                  const categorySubcategories = materialNodeSubcategoryOrder[category].filter((subcategory) =>
                    availableNodes.some(
                      (definition) => definition.category === category && definition.subcategory === subcategory,
                    ),
                  );
                  const submenuDirection =
                    addMenu && canvasRef.current && addMenu.screen[0] + 280 + 2 * 240 > canvasRef.current.clientWidth
                      ? 'left'
                      : 'right';

                  return (
                    <div
                      className={`material-node-menu-cascade-entry material-node-menu-cascade-${submenuDirection}`}
                      key={category}
                      onMouseEnter={() => {
                        setNodeMenuCategory(category);
                        setNodeMenuSubcategory(null);
                      }}
                    >
                      <UiContextMenuItem
                        aria-expanded={categoryActive}
                        aria-haspopup="menu"
                        onClick={() => {
                          setNodeMenuCategory(category);
                          setNodeMenuSubcategory(null);
                        }}
                        trailing={<ChevronRight size={13} />}
                      >
                        <strong>{category}</strong>
                      </UiContextMenuItem>
                      {categoryActive && (
                        <UiContextMenu
                          aria-label={`${category} material node categories`}
                          className="material-node-menu-submenu"
                          maxHeight={380}
                          width={240}
                        >
                          {categorySubcategories.map((subcategory) => {
                            const subcategoryActive = nodeMenuSubcategory === subcategory;
                            const subcategoryNodes = availableNodes.filter(
                              (definition) =>
                                definition.category === category && definition.subcategory === subcategory,
                            );
                            return (
                              <div
                                className={`material-node-menu-cascade-entry material-node-menu-cascade-${submenuDirection}`}
                                key={subcategory}
                                onMouseEnter={() => setNodeMenuSubcategory(subcategory)}
                              >
                                <UiContextMenuItem
                                  aria-expanded={subcategoryActive}
                                  aria-haspopup="menu"
                                  onClick={() => setNodeMenuSubcategory(subcategory)}
                                  trailing={<ChevronRight size={13} />}
                                >
                                  <strong>{subcategory}</strong>
                                </UiContextMenuItem>
                                {subcategoryActive && (
                                  <UiContextMenu
                                    aria-label={`${subcategory} material nodes`}
                                    className="material-node-menu-submenu"
                                    maxHeight={380}
                                    width={240}
                                  >
                                    {subcategoryNodes.map((definition) => (
                                      <UiContextMenuItem key={definition.type} onClick={() => addNode(definition.type)}>
                                        <strong>{definition.title}</strong>
                                      </UiContextMenuItem>
                                    ))}
                                  </UiContextMenu>
                                )}
                              </div>
                            );
                          })}
                        </UiContextMenu>
                      )}
                    </div>
                  );
                })}
          </div>
        </UiContextMenu>
      )}
    </div>
  );
}
