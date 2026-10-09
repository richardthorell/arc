import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ChevronRight, Copy, Magnet, Plus, RotateCcw, Scan, Search, Trash2, WandSparkles } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import { AssetPicker, type AssetPickerItem } from '../inspector/AssetPicker';
import {
  GraphDiagnosticBadge,
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
  summarizeGraphDiagnostics,
  type GraphDiagnostic,
  type GraphPoint,
  type GraphSelection,
} from '../graph';
import {
  UiButton,
  UiColorControl,
  UiContextMenu,
  UiContextMenuItem,
  UiIconButton,
  UiNodeCard,
  UiSelect,
  UiSlider,
  UiTextInput,
  type UiColorValue,
} from '../ui';
import { materialGraphDomain } from './materialGraphDomain';
import { materialFunctionCompatibleWithSlot } from './materialInstanceAuthoring';
import {
  clampMaterialScalarValue,
  cloneMaterialGraph,
  createMaterialNode,
  materialGraphId,
  isMaterialTextureSampleNodeType,
  materialScalarRange,
  materialNodeCategoryOrder,
  materialNodeSubcategoryOrder,
  type MaterialFunctionAssetJson,
  type MaterialGraph,
  type MaterialGraphConnection,
  type MaterialGraphNode,
  type MaterialGraphNodeType,
  type MaterialGraphPinRef,
  type MaterialNodeCategory,
  type MaterialNodeSubcategory,
} from './materialGraphTypes';
import {
  redoMaterialGraph,
  replaceMaterialGraph,
  replaceMaterialGraphViewport,
  undoMaterialGraph,
} from './materialDocumentState';
import {
  autoArrangeMaterialGraph,
  frameMaterialGraphViewport,
  materialNodeHeight,
  materialNodeWidth,
  snapMaterialGraphPoint,
} from './materialGraphLayout';
import { MaterialTextureSampleEditor } from './MaterialTextureSampleEditor';

const headerHeight = 34;
const pinRowHeight = 25;
const nodePaddingTop = 9;

type AddMenuCategory = Exclude<MaterialNodeCategory, 'Output'>;

type MaterialSubmenuAnchor = {
  left: number;
  right: number;
  top: number;
};

const materialSubmenuWidth = 240;
const materialSubmenuMaxHeight = 380;
const materialSubmenuMargin = 8;
const materialSubmenuOverlap = 2;

const materialSubmenuAnchor = (element: HTMLElement): MaterialSubmenuAnchor => {
  const rect = element.getBoundingClientRect();
  return { left: rect.left, right: rect.right, top: rect.top };
};

const materialSubmenuPosition = (anchor: MaterialSubmenuAnchor, itemCount: number) => {
  const estimatedHeight = Math.min(materialSubmenuMaxHeight, Math.max(40, itemCount * 32 + 8));
  const top = Math.max(
    materialSubmenuMargin,
    Math.min(anchor.top - 4, window.innerHeight - estimatedHeight - materialSubmenuMargin),
  );
  const roomOnRight =
    anchor.right - materialSubmenuOverlap + materialSubmenuWidth + materialSubmenuMargin <= window.innerWidth;
  const left = roomOnRight
    ? Math.max(
        materialSubmenuMargin,
        Math.min(
          anchor.right - materialSubmenuOverlap,
          window.innerWidth - materialSubmenuWidth - materialSubmenuMargin,
        ),
      )
    : Math.max(materialSubmenuMargin, anchor.left - materialSubmenuWidth + materialSubmenuOverlap);
  return {
    left,
    top,
    maxHeight: Math.max(80, Math.min(materialSubmenuMaxHeight, window.innerHeight - top - materialSubmenuMargin)),
  };
};

const editableValueNode = (node: MaterialGraphNode) =>
  node.type === 'constant' ||
  node.type === 'vector2' ||
  node.type === 'vector3' ||
  node.type === 'vector4' ||
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
const colorValue = (value: unknown): UiColorValue => {
  const components = Array.isArray(value) ? value : [];
  return {
    x: colorChannel(components[0] ?? 1),
    y: colorChannel(components[1] ?? 1),
    z: colorChannel(components[2] ?? 1),
    w: colorChannel(components[3] ?? 1),
  };
};

type MaterialFunctionAssetOption = {
  asset: AssetPickerItem;
  document: MaterialFunctionAssetJson;
};

const materialFunctionReferencePath = (path: string, scope: 'builtin' | 'project') =>
  scope === 'builtin' ? path.replaceAll('\\', '/').replace(/^builtin\//i, '') : path;

const functionReferences = (node: MaterialGraphNode) =>
  Array.isArray(node.values.functions)
    ? node.values.functions.flatMap((reference) => {
        if (!reference || typeof reference !== 'object') return [];
        const path = (reference as { path?: unknown }).path;
        return typeof path === 'string' && path.trim() ? [{ path }] : [];
      })
    : [];

function MaterialFunctionReferenceEditor({
  node,
  readOnly,
  onChange,
}: {
  node: MaterialGraphNode;
  readOnly: boolean;
  onChange: (node: MaterialGraphNode) => void;
}) {
  const [functions, setFunctions] = useState<MaterialFunctionAssetOption[]>([]);

  useEffect(() => {
    let cancelled = false;
    void window.arc.host
      .query('project.assets')
      .then(async (response: unknown) => {
        if (cancelled || !response || typeof response !== 'object') return;
        const payload = (response as { payload?: { assets?: Array<Record<string, unknown>> } }).payload;
        const candidates = (payload?.assets ?? []).flatMap((asset) => {
          if (asset.kind !== 'materialFunction') return [];
          const path = typeof asset.path === 'string' ? asset.path : '';
          const sourcePath = typeof asset.sourcePath === 'string' ? asset.sourcePath : path;
          if (!path && !sourcePath) return [];
          const scope = asset.scope === 'builtin' ? ('builtin' as const) : ('project' as const);
          const authoringPath = scope === 'builtin' ? path : sourcePath || path;
          return [
            {
              asset: {
                id: typeof asset.guid === 'string' && asset.guid ? asset.guid : authoringPath,
                guid: typeof asset.guid === 'string' ? asset.guid : undefined,
                typeId: typeof asset.typeId === 'string' ? asset.typeId : undefined,
                name:
                  typeof asset.title === 'string' && asset.title.trim()
                    ? asset.title
                    : (authoringPath.split('/').at(-1) ?? authoringPath),
                title: typeof asset.title === 'string' ? asset.title : undefined,
                path: materialFunctionReferencePath(authoringPath, scope),
                sourcePath,
                kind: 'materialFunction',
                status:
                  asset.state === 'ready' ||
                  asset.state === 'stale' ||
                  asset.state === 'failed' ||
                  asset.state === 'importing'
                    ? asset.state
                    : 'unknown',
                scope,
                readOnly: Boolean(asset.readOnly),
              } satisfies AssetPickerItem,
              scope,
              authoringPath,
            },
          ];
        });
        const loaded = (
          await Promise.all(
            candidates.map(async (candidate): Promise<MaterialFunctionAssetOption | null> => {
              try {
                const file = await window.arc.projects.readText(candidate.authoringPath, candidate.scope);
                const document = JSON.parse(file.text) as MaterialFunctionAssetJson;
                if (document.kind !== 'materialFunction' || document.version !== 1) return null;
                return { asset: candidate.asset, document };
              } catch {
                return null;
              }
            }),
          )
        ).filter((option): option is MaterialFunctionAssetOption => option !== null);
        loaded.sort((left, right) => left.document.name.localeCompare(right.document.name));
        if (!cancelled) setFunctions(loaded);
      })
      .catch(() => {
        if (!cancelled) setFunctions([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const references = functionReferences(node);
  const selectedPath = typeof node.values.path === 'string' ? node.values.path : '';
  const inputPins = Array.isArray(node.values.inputPins) ? node.values.inputPins : [];
  const outputPins = Array.isArray(node.values.outputPins) ? node.values.outputPins : [];
  const selectable = node.type === 'functionCall';
  const selectedDocument = functions.find((option) => option.asset.path === selectedPath)?.document;
  const contractInputs =
    inputPins.length > 0
      ? inputPins
      : (selectedDocument?.inputs ??
        functions.find((option) => option.asset.path === references[0]?.path)?.document.inputs ??
        []);
  const contractOutputs =
    outputPins.length > 0
      ? outputPins
      : (selectedDocument?.outputs ??
        functions.find((option) => option.asset.path === references[0]?.path)?.document.outputs ??
        []);

  if (!selectable) {
    const options = [
      { value: '', label: 'Select function…' },
      ...functions.map((option) => ({ value: option.asset.path, label: option.document.name })),
    ];
    return (
      <div className="material-node-inline-value">
        <input
          aria-label="Function Slot name"
          disabled={readOnly}
          value={typeof node.values.name === 'string' ? node.values.name : 'Function Slot'}
          onChange={(event) => onChange({ ...node, values: { ...node.values, name: event.target.value } })}
        />
        <UiSelect
          ariaLabel="Default Material Function"
          disabled={readOnly}
          options={options}
          value={selectedPath}
          onValueChange={(value) => {
            const option = functions.find((candidate) => candidate.asset.path === value);
            if (!option) return;
            onChange({
              ...node,
              values: {
                ...node.values,
                path: option.asset.path,
                functionName: option.document.name,
                inputPins: option.document.inputs,
                outputPins: option.document.outputs,
              },
            });
          }}
        />
      </div>
    );
  }

  const referenced: Array<{
    reference: { path: string };
    option: MaterialFunctionAssetOption | null;
  }> = references.map((reference) => ({
    reference,
    option: functions.find((candidate) => candidate.asset.path === reference.path) ?? null,
  }));
  const referencedPaths = new Set(references.map((reference) => reference.path));
  const pickerAssets = functions
    .filter((option) => !referencedPaths.has(option.asset.path))
    .map((option) => option.asset);
  const compatibility = (asset: AssetPickerItem) => {
    const option = functions.find((candidate) => candidate.asset.path === asset.path);
    if (!option) return 'Material Function metadata is unavailable';
    return materialFunctionCompatibleWithSlot(
      contractInputs as MaterialFunctionAssetJson['inputs'],
      contractOutputs as MaterialFunctionAssetJson['outputs'],
      option.document,
    )
      ? null
      : 'Function signature does not match this Function Call';
  };

  const setActive = (path: string, document?: MaterialFunctionAssetJson) => {
    onChange({
      ...node,
      values: {
        ...node.values,
        path,
        functionName: document?.name ?? node.values.functionName,
        ...(contractInputs.length === 0 && document ? { inputPins: document.inputs } : {}),
        ...(contractOutputs.length === 0 && document ? { outputPins: document.outputs } : {}),
      },
    });
  };

  const addFunction = (path: string) => {
    const option = functions.find((candidate) => candidate.asset.path === path);
    if (!option || compatibility(option.asset)) return;
    const nextReferences = [...references, { path: option.asset.path }];
    const first = references.length === 0;
    onChange({
      ...node,
      values: {
        ...node.values,
        functions: nextReferences,
        ...(first
          ? {
              path: option.asset.path,
              functionName: option.document.name,
              inputPins: option.document.inputs,
              outputPins: option.document.outputs,
            }
          : {}),
      },
    });
  };

  const removeFunction = (path: string) => {
    const nextReferences = references.filter((reference) => reference.path !== path);
    const removedActive = selectedPath === path;
    const nextActive = removedActive ? (nextReferences[0]?.path ?? '') : selectedPath;
    const nextDocument = functions.find((option) => option.asset.path === nextActive)?.document;
    onChange({
      ...node,
      values: {
        ...node.values,
        functions: nextReferences,
        path: nextActive,
        functionName: nextDocument?.name ?? '',
        ...(nextReferences.length === 0 ? { inputPins: [], outputPins: [] } : {}),
      },
    });
  };

  return (
    <div className="material-function-call-editor">
      <UiTextInput
        aria-label="Function Call name"
        disabled={readOnly}
        value={typeof node.values.name === 'string' ? node.values.name : 'Material Function'}
        onChange={(event) => onChange({ ...node, values: { ...node.values, name: event.target.value } })}
      />
      <div className="material-function-call-list">
        {referenced.map(({ reference, option }) => {
          const active = reference.path === selectedPath;
          const label = option?.document.name ?? reference.path.split('/').at(-1) ?? reference.path;
          return (
            <div className={`material-function-call-entry${active ? ' is-active' : ''}`} key={reference.path}>
              <button
                aria-label={`Use ${label}`}
                aria-pressed={active}
                className="material-function-call-select"
                disabled={readOnly}
                onClick={() => setActive(reference.path, option?.document)}
                type="button"
              >
                <span className="material-function-call-radio" aria-hidden="true" />
                <span>{label}</span>
              </button>
              {!readOnly && references.length > 1 && (
                <UiIconButton
                  label={`Remove ${label}`}
                  onClick={() => removeFunction(reference.path)}
                  title={`Remove ${label}`}
                >
                  <Trash2 size={12} />
                </UiIconButton>
              )}
            </div>
          );
        })}
        {!readOnly && (
          <div className="material-function-call-add">
            <AssetPicker
              allowEmpty={false}
              assetCompatibility={compatibility}
              assetKinds={['materialFunction']}
              assets={pickerAssets}
              assetTypeLabel="Material Function"
              label="Material Function"
              showLabel={false}
              triggerMode="add"
              value=""
              onChange={addFunction}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function MaterialNodeValueEditor({
  node,
  readOnly,
  onChange,
}: {
  node: MaterialGraphNode;
  readOnly: boolean;
  onChange: (node: MaterialGraphNode) => void;
}) {
  if (node.type === 'constant') {
    const value = typeof node.values.value === 'number' && Number.isFinite(node.values.value) ? node.values.value : 0;
    const range = materialScalarRange(node);
    const setValue = (nextValue: number) => onChange(nextNodeValue(node, clampMaterialScalarValue(nextValue, range)));
    const setRange = (min: number, max: number) => {
      const nextRange = { min: Math.min(min, max), max: Math.max(min, max) };
      onChange({
        ...node,
        values: {
          ...node.values,
          ...nextRange,
          value: clampMaterialScalarValue(value, nextRange),
        },
      });
    };

    return (
      <div className="material-node-scalar-value">
        <label className="material-node-inline-value">
          Value
          <input
            aria-label="Scalar value"
            disabled={readOnly}
            max={range?.max}
            min={range?.min}
            type="number"
            step="0.01"
            value={value}
            onChange={(event) => setValue(Number(event.target.value))}
          />
        </label>
        {range && (
          <UiSlider
            aria-label="Scalar range value"
            disabled={readOnly}
            min={range.min}
            max={range.max}
            step={Math.max(0.001, (range.max - range.min) / 100)}
            value={value}
            onValueChange={setValue}
          />
        )}
        <label className="material-node-range-toggle">
          <input
            checked={Boolean(range)}
            disabled={readOnly}
            type="checkbox"
            onChange={(event) => {
              if (event.target.checked) {
                const nextRange = { min: 0, max: 1 };
                onChange({
                  ...node,
                  values: {
                    ...node.values,
                    ...nextRange,
                    value: clampMaterialScalarValue(value, nextRange),
                  },
                });
                return;
              }
              const values = { ...node.values };
              delete values.min;
              delete values.max;
              onChange({ ...node, values });
            }}
          />
          <span>Range</span>
        </label>
        {range && (
          <div className="material-node-range-bounds">
            <label>
              Min
              <input
                aria-label="Scalar minimum"
                disabled={readOnly}
                type="number"
                step="0.01"
                value={range.min}
                onChange={(event) => setRange(Number(event.target.value), range.max)}
              />
            </label>
            <label>
              Max
              <input
                aria-label="Scalar maximum"
                disabled={readOnly}
                type="number"
                step="0.01"
                value={range.max}
                onChange={(event) => setRange(range.min, Number(event.target.value))}
              />
            </label>
          </div>
        )}
      </div>
    );
  }

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

  if (node.type === 'colorRgba') {
    const color = colorValue(node.values.value);
    return (
      <UiColorControl
        allowAlpha
        label="Color"
        onCommit={(next) => {
          if (readOnly) return;
          onChange(nextNodeValue(node, [next.x, next.y, next.z, next.w]));
        }}
        value={color}
      />
    );
  }

  if (isMaterialTextureSampleNodeType(node.type))
    return <MaterialTextureSampleEditor node={node} readOnly={readOnly} onChange={onChange} />;

  if (node.type === 'functionCall' || node.type === 'functionSlot')
    return <MaterialFunctionReferenceEditor node={node} readOnly={readOnly} onChange={onChange} />;

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

export function MaterialGraphEditor({
  document,
  graph,
  loaded = true,
  showGrid = true,
  dimUnrelated = false,
  onGraphChange,
  onViewportChange,
  onUndo,
  onRedo,
  diagnostics = [],
}: {
  document: EditorDocument;
  graph: MaterialGraph;
  loaded?: boolean;
  showGrid?: boolean;
  dimUnrelated?: boolean;
  onGraphChange?: (graph: MaterialGraph, options?: { recordHistory?: boolean; message?: string }) => void;
  onViewportChange?: (viewport: NonNullable<MaterialGraph['viewport']>) => void;
  onUndo?: () => void;
  onRedo?: () => void;
  diagnostics?: GraphDiagnostic[];
}) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const invalidConnectionNodeRef = useRef<HTMLElement | null>(null);
  const invalidConnectionTimeoutRef = useRef<number | null>(null);
  const autoFramedDocumentRef = useRef<string | null>(null);
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
  const [categoryMenuAnchor, setCategoryMenuAnchor] = useState<MaterialSubmenuAnchor | null>(null);
  const [subcategoryMenuAnchor, setSubcategoryMenuAnchor] = useState<MaterialSubmenuAnchor | null>(null);
  const [snapEnabled, setSnapEnabled] = useState(true);
  const commitGraph = useCallback(
    (next: MaterialGraph, options: { recordHistory?: boolean; message?: string } = {}) => {
      if (onGraphChange) onGraphChange(next, options);
      else replaceMaterialGraph(document, next, options);
    },
    [document, onGraphChange],
  );
  const commitViewport = useCallback(
    (next: NonNullable<MaterialGraph['viewport']>) => {
      if (onViewportChange) onViewportChange(next);
      else replaceMaterialGraphViewport(document, next);
    },
    [document, onViewportChange],
  );
  const undo = useCallback(() => {
    if (onUndo) onUndo();
    else undoMaterialGraph(document);
  }, [document, onUndo]);
  const redo = useCallback(() => {
    if (onRedo) onRedo();
    else redoMaterialGraph(document);
  }, [document, onRedo]);
  const viewport = useMemo(() => graph.viewport ?? { x: 40, y: 40, zoom: 1 }, [graph.viewport]);
  const diagnosticSummaries = useMemo(
    () => new Map(summarizeGraphDiagnostics(diagnostics).map((summary) => [summary.nodeId, summary])),
    [diagnostics],
  );
  const relatedNodeIds = useMemo(() => {
    if (!dimUnrelated || selectedNodes.size === 0) return null;
    const related = new Set(selectedNodes);
    let changed = true;
    while (changed) {
      changed = false;
      for (const connection of graph.connections) {
        if (!related.has(connection.from.nodeId) && !related.has(connection.to.nodeId)) continue;
        if (!related.has(connection.from.nodeId)) {
          related.add(connection.from.nodeId);
          changed = true;
        }
        if (!related.has(connection.to.nodeId)) {
          related.add(connection.to.nodeId);
          changed = true;
        }
      }
    }
    return related;
  }, [dimUnrelated, graph.connections, selectedNodes]);

  useEffect(() => {
    setSelectedNodes((current) => new Set([...current].filter((id) => graph.nodes.some((node) => node.id === id))));
  }, [graph.nodes]);

  useEffect(
    () => () => {
      if (invalidConnectionTimeoutRef.current !== null) window.clearTimeout(invalidConnectionTimeoutRef.current);
    },
    [],
  );

  const flashRejectedConnection = useCallback((nodeId: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const targetNode = Array.from(canvas.querySelectorAll<HTMLElement>('[data-node-id]')).find(
      (candidate) => candidate.dataset.nodeId === nodeId,
    );
    if (!targetNode) return;

    if (invalidConnectionTimeoutRef.current !== null) window.clearTimeout(invalidConnectionTimeoutRef.current);
    invalidConnectionNodeRef.current?.classList.remove('is-connection-invalid');
    targetNode.classList.remove('is-connection-invalid');
    void targetNode.offsetWidth;
    targetNode.classList.add('is-connection-invalid');
    invalidConnectionNodeRef.current = targetNode;
    invalidConnectionTimeoutRef.current = window.setTimeout(() => {
      targetNode.classList.remove('is-connection-invalid');
      if (invalidConnectionNodeRef.current === targetNode) invalidConnectionNodeRef.current = null;
      invalidConnectionTimeoutRef.current = null;
    }, 1000);
  }, []);

  const mutate = useCallback(
    (updater: (draft: MaterialGraph) => void, recordHistory = true) => {
      if (document.readOnly) return;
      const next = cloneMaterialGraph(graph);
      updater(next);
      commitGraph(next, { recordHistory });
    },
    [commitGraph, document, graph],
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
    (patch: Partial<typeof viewport>) =>
      commitViewport({
        ...viewport,
        ...patch,
      }),
    [commitViewport, viewport],
  );

  const frameAll = useCallback(() => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0 || graph.nodes.length === 0) return false;
    commitViewport(frameMaterialGraphViewport(graph, rect.width, rect.height));
    return true;
  }, [commitViewport, graph]);

  const focusDiagnosticNode = useCallback(
    (nodeId: string) => {
      const node = graph.nodes.find((candidate) => candidate.id === nodeId);
      if (!node) return;
      setSelectedNodes(new Set([nodeId]));
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect || rect.width <= 0 || rect.height <= 0) return;
      commitViewport({
        x: rect.width / 2 - (node.position[0] + materialNodeWidth(node.type) / 2) * viewport.zoom,
        y: rect.height / 2 - (node.position[1] + materialNodeHeight(node) / 2) * viewport.zoom,
        zoom: viewport.zoom,
      });
    },
    [commitViewport, graph.nodes, viewport.zoom],
  );

  const setZoomAroundCenter = useCallback(
    (requestedZoom: number) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect || rect.width <= 0 || rect.height <= 0) return;
      const zoom = clampGraphZoom(requestedZoom);
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const graphCenterX = (centerX - viewport.x) / viewport.zoom;
      const graphCenterY = (centerY - viewport.y) / viewport.zoom;
      updateViewport({
        x: centerX - graphCenterX * zoom,
        y: centerY - graphCenterY * zoom,
        zoom,
      });
    },
    [updateViewport, viewport],
  );

  const autoArrange = useCallback(() => {
    if (document.readOnly) return;
    const arranged = autoArrangeMaterialGraph(graph);
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect && rect.width > 0 && rect.height > 0)
      arranged.viewport = frameMaterialGraphViewport(arranged, rect.width, rect.height);
    commitGraph(arranged, { message: 'Auto-arranged material graph' });
  }, [commitGraph, document, graph]);

  useLayoutEffect(() => {
    if (!loaded || autoFramedDocumentRef.current === document.id || graph.nodes.length === 0) return;
    const frame = window.requestAnimationFrame(() => {
      if (frameAll()) autoFramedDocumentRef.current = document.id;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [document.id, frameAll, graph.nodes.length, loaded]);

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
            if (origin) {
              const position: GraphPoint = [origin[0] + deltaX, origin[1] + deltaY];
              node.position = snapEnabled ? snapMaterialGraphPoint(position) : position;
            }
          }
        }, false);
      } else if (pan) {
        updateViewport({
          x: pan.viewport[0] + (event.clientX - pan.start[0]),
          y: pan.viewport[1] + (event.clientY - pan.start[1]),
        });
      } else if (box) {
        setBox({ ...box, current: point });
      }
    };
    const up = () => {
      if (drag) commitGraph(graph, { recordHistory: true });
      if (box) {
        const bounds = graphSelectionBounds(box);
        setSelectedNodes(
          new Set(
            graph.nodes
              .filter(
                (node) =>
                  node.position[0] + materialNodeWidth(node.type) >= bounds.left &&
                  node.position[0] <= bounds.right &&
                  node.position[1] + materialNodeHeight(node) >= bounds.top &&
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
  }, [box, commitGraph, document, drag, graph, graphPoint, mutate, pan, snapEnabled, updateViewport]);

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
      const position: GraphPoint = [source.position[0] + 36, source.position[1] + 36];
      return { ...source, id, position: snapEnabled ? snapMaterialGraphPoint(position) : position };
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
        if (event.shiftKey) redo();
        else undo();
      } else if (command && event.key.toLocaleLowerCase() === 'y') {
        event.preventDefault();
        redo();
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
      flashRejectedConnection(target.nodeId);
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
    setCategoryMenuAnchor(null);
    setSubcategoryMenuAnchor(null);
  };

  const openCategoryMenu = (category: AddMenuCategory, element: HTMLElement) => {
    setNodeMenuCategory(category);
    setNodeMenuSubcategory(null);
    setCategoryMenuAnchor(materialSubmenuAnchor(element));
    setSubcategoryMenuAnchor(null);
  };

  const openSubcategoryMenu = (subcategory: MaterialNodeSubcategory, element: HTMLElement) => {
    setNodeMenuSubcategory(subcategory);
    setSubcategoryMenuAnchor(materialSubmenuAnchor(element));
  };

  const addNode = (type: MaterialGraphNodeType) => {
    if (type === 'output' || document.readOnly || !addMenu) return;
    const node = createMaterialNode(type, snapEnabled ? snapMaterialGraphPoint(addMenu.graph) : addMenu.graph);
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
      .filter((definition) => definition.type !== 'output')
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
      className={[
        'material-graph-canvas',
        document.readOnly ? 'read-only' : '',
        showGrid ? '' : 'hide-grid',
        dimUnrelated ? 'dim-unrelated' : '',
      ]
        .filter(Boolean)
        .join(' ')}
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
        updateViewport({ x, y, zoom });
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
      </div>

      <div
        className="material-graph-navigation-toolbar"
        onPointerDown={(event) => event.stopPropagation()}
        onWheel={(event) => event.stopPropagation()}
      >
        <UiButton onClick={() => frameAll()} title="Frame all nodes" variant="toolbar">
          <Scan size={13} /> Frame All
        </UiButton>
        <UiButton
          disabled={document.readOnly}
          onClick={autoArrange}
          title="Arrange nodes by connection flow"
          variant="toolbar"
        >
          <WandSparkles size={13} /> Arrange
        </UiButton>
        <span className="material-graph-toolbar-divider" />
        <label className="material-graph-zoom-control">
          <span>Zoom</span>
          <UiSlider
            aria-label="Material graph zoom"
            max={180}
            min={35}
            step={5}
            value={Math.round(viewport.zoom * 100)}
            onValueChange={(value) => setZoomAroundCenter(value / 100)}
          />
          <output>{Math.round(viewport.zoom * 100)}%</output>
        </label>
        <UiIconButton label="Reset zoom to 100%" onClick={() => setZoomAroundCenter(1)}>
          <RotateCcw size={13} />
        </UiIconButton>
        <span className="material-graph-toolbar-divider" />
        <UiButton
          active={snapEnabled}
          aria-pressed={snapEnabled}
          onClick={() => setSnapEnabled((enabled) => !enabled)}
          title="Snap nodes to the 20 px graph grid"
          variant="toolbar"
        >
          <Magnet size={13} /> Snap
        </UiButton>
      </div>

      <GraphViewportLayer className="material-graph-transform" viewport={viewport}>
        <GraphWireLayer className="material-graph-wires" pendingPath={pendingPath} wires={wirePaths} />
        {graph.nodes.map((node) => {
          const definition = materialGraphDomain.getNodeDefinition(node);
          const selected = selectedNodes.has(node.id);
          const diagnosticSummary = diagnosticSummaries.get(node.id);
          return (
            <UiNodeCard
              badge={node.parameter?.exposed ? 'P' : undefined}
              badgeTitle={node.parameter?.exposed ? `Parameter: ${node.parameter.name}` : undefined}
              className={[
                'material-graph-node',
                `material-graph-node-${node.type}`,
                relatedNodeIds && !relatedNodeIds.has(node.id) ? 'is-unrelated' : '',
              ]
                .filter(Boolean)
                .join(' ')}
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
              {diagnosticSummary && (
                <GraphDiagnosticBadge summary={diagnosticSummary} onActivate={() => focusDiagnosticNode(node.id)} />
              )}
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
                        title={
                          pin.semanticRange
                            ? `${pin.label} · ${pin.type} · expected ${pin.semanticRange.min}..${pin.semanticRange.max}`
                            : `${pin.label} · ${pin.type}`
                        }
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
                  const categoryPosition =
                    categoryActive && categoryMenuAnchor
                      ? materialSubmenuPosition(categoryMenuAnchor, categorySubcategories.length)
                      : null;

                  return (
                    <div
                      className="material-node-menu-cascade-entry"
                      key={category}
                      onMouseEnter={(event) => openCategoryMenu(category, event.currentTarget)}
                    >
                      <UiContextMenuItem
                        aria-expanded={categoryActive}
                        aria-haspopup="menu"
                        onClick={(event) => openCategoryMenu(category, event.currentTarget)}
                        trailing={<ChevronRight size={13} />}
                      >
                        <strong>{category}</strong>
                      </UiContextMenuItem>
                      {categoryActive &&
                        categoryPosition &&
                        createPortal(
                          <UiContextMenu
                            aria-label={`${category} material node categories`}
                            className="material-node-menu-submenu material-node-menu-submenu-level-1"
                            maxHeight={categoryPosition.maxHeight}
                            style={{ position: 'fixed' }}
                            width={materialSubmenuWidth}
                            x={categoryPosition.left}
                            y={categoryPosition.top}
                          >
                            {categorySubcategories.map((subcategory) => {
                              const subcategoryActive = nodeMenuSubcategory === subcategory;
                              const subcategoryNodes = availableNodes.filter(
                                (definition) =>
                                  definition.category === category && definition.subcategory === subcategory,
                              );
                              const subcategoryPosition =
                                subcategoryActive && subcategoryMenuAnchor
                                  ? materialSubmenuPosition(subcategoryMenuAnchor, subcategoryNodes.length)
                                  : null;
                              return (
                                <div
                                  className="material-node-menu-cascade-entry"
                                  key={subcategory}
                                  onMouseEnter={(event) => openSubcategoryMenu(subcategory, event.currentTarget)}
                                >
                                  <UiContextMenuItem
                                    aria-expanded={subcategoryActive}
                                    aria-haspopup="menu"
                                    onClick={(event) => openSubcategoryMenu(subcategory, event.currentTarget)}
                                    trailing={<ChevronRight size={13} />}
                                  >
                                    <strong>{subcategory}</strong>
                                  </UiContextMenuItem>
                                  {subcategoryActive &&
                                    subcategoryPosition &&
                                    createPortal(
                                      <UiContextMenu
                                        aria-label={`${subcategory} material nodes`}
                                        className="material-node-menu-submenu material-node-menu-submenu-level-2"
                                        maxHeight={subcategoryPosition.maxHeight}
                                        style={{ position: 'fixed' }}
                                        width={materialSubmenuWidth}
                                        x={subcategoryPosition.left}
                                        y={subcategoryPosition.top}
                                      >
                                        {subcategoryNodes.map((definition) => (
                                          <UiContextMenuItem
                                            key={definition.type}
                                            onClick={() => addNode(definition.type)}
                                          >
                                            <strong>{definition.title}</strong>
                                          </UiContextMenuItem>
                                        ))}
                                      </UiContextMenu>,
                                      globalThis.document.body,
                                    )}
                                </div>
                              );
                            })}
                          </UiContextMenu>,
                          globalThis.document.body,
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
