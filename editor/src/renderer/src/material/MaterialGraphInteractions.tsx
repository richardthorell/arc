import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';

import type { EditorDocument } from '../editors/editorTypes';
import { graphConnectionPath, graphPinKey, type GraphPoint } from '../graph';
import { materialGraphDomain } from './materialGraphDomain';
import { MaterialGraphEditor } from './MaterialGraphEditor';
import type { MaterialGraph, MaterialGraphNode, MaterialGraphPinType, MaterialNodePin } from './materialGraphTypes';
import './materialGraphInteractions.css';

type MaterialPinMetadata = {
  direction: 'input' | 'output';
  key: string;
  node: MaterialGraphNode;
  pin: MaterialNodePin;
  tooltip: string;
};

type MaterialWireOverlay = {
  fromPinKey: string;
  id: string;
  label: string;
  path: string;
  toPinKey: string;
};

type HoveredTooltip = {
  id: string;
  position: GraphPoint;
  text: string;
};

const materialPinTypeLabel = (type: MaterialGraphPinType) => {
  switch (type) {
    case 'float':
      return 'Float';
    case 'vec2':
      return 'Vector2';
    case 'vec3':
      return 'Vector3';
    case 'vec4':
      return 'Vector4';
    case 'texture2d':
      return 'Texture2D';
    case 'numeric':
      return 'Numeric';
  }
};

const materialPinDescriptions: Record<string, string> = {
  'output:baseColor': 'surface albedo',
  'output:metallic': 'metalness response',
  'output:roughness': 'microsurface roughness',
  'output:normal': 'surface normal',
  'output:clearCoatNormal': 'clear-coat normal',
  'output:tangent': 'surface tangent direction',
  'output:ao': 'ambient occlusion',
  'output:emissive': 'emitted light color',
  'output:opacity': 'surface opacity',
  'output:alphaClip': 'cutout threshold',
  'output:indexOfRefraction': 'index of refraction',
  'output:clearCoat': 'clear-coat amount',
  'output:clearCoatRoughness': 'clear-coat roughness',
  'output:sheen': 'sheen amount',
  'output:sheenColor': 'sheen tint',
  'output:sheenRoughness': 'sheen roughness',
  'output:anisotropy': 'directional highlight amount',
  'output:anisotropyRotation': 'anisotropy orientation',
  'output:transmission': 'transmitted light amount',
  'output:thickness': 'transmission thickness',
  'output:attenuationColor': 'transmission attenuation tint',
  'output:attenuationDistance': 'attenuation distance',
  'output:subsurfaceColor': 'subsurface scattering tint',
  'output:subsurface': 'subsurface amount',
  'textureSample:uv': 'texture coordinates',
  'texCoord:uv': 'mesh texture coordinates',
  'time:seconds': 'elapsed material time',
  'normalMap:texture': 'encoded tangent-space normal',
  'normalMap:normal': 'decoded surface normal',
};

const materialPinTooltip = (metadata: Omit<MaterialPinMetadata, 'key' | 'tooltip'>) => {
  const direction = metadata.direction === 'input' ? 'Input' : 'Output';
  const type = materialPinTypeLabel(metadata.pin.type);
  const description = materialPinDescriptions[`${metadata.node.type}:${metadata.pin.id}`];
  return `${direction} · ${type} • ${metadata.pin.label}${description ? ` — ${description}` : ''}`;
};

const localPointerPosition = (host: HTMLElement, clientX: number, clientY: number): GraphPoint => {
  const rect = host.getBoundingClientRect();
  return [clientX - rect.left, clientY - rect.top];
};

const sameWireOverlays = (left: MaterialWireOverlay[], right: MaterialWireOverlay[]) =>
  left.length === right.length &&
  left.every((wire, index) => {
    const candidate = right[index];
    return (
      candidate !== undefined &&
      wire.id === candidate.id &&
      wire.path === candidate.path &&
      wire.label === candidate.label &&
      wire.fromPinKey === candidate.fromPinKey &&
      wire.toPinKey === candidate.toPinKey
    );
  });

const graphPinElementMap = (host: HTMLElement) =>
  new Map(
    Array.from(host.querySelectorAll<HTMLElement>('[data-graph-pin-key]')).flatMap((element) => {
      const key = element.dataset.graphPinKey;
      return key ? [[key, element] as const] : [];
    }),
  );

const pinSocketCenter = (element: HTMLElement, hostRect: DOMRect): GraphPoint | null => {
  const socket = element.querySelector<HTMLElement>('[data-graph-pin-socket]');
  if (!socket) return null;
  const rect = socket.getBoundingClientRect();
  return [rect.left + rect.width / 2 - hostRect.left, rect.top + rect.height / 2 - hostRect.top];
};

export function materialConnectionFlowIds(graph: MaterialGraph, connectionId: string): Set<string> {
  const selected = graph.connections.find((connection) => connection.id === connectionId);
  if (!selected) return new Set();

  const flow = new Set<string>([selected.id]);
  const visitedNodes = new Set<string>();
  const pendingNodes = [selected.from.nodeId];

  while (pendingNodes.length > 0) {
    const nodeId = pendingNodes.pop();
    if (!nodeId || visitedNodes.has(nodeId)) continue;
    visitedNodes.add(nodeId);

    for (const connection of graph.connections) {
      if (connection.to.nodeId !== nodeId || flow.has(connection.id)) continue;
      flow.add(connection.id);
      pendingNodes.push(connection.from.nodeId);
    }
  }

  return flow;
}

export function MaterialGraphWithInteractions({ document, graph }: { document: EditorDocument; graph: MaterialGraph }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [wires, setWires] = useState<MaterialWireOverlay[]>([]);
  const [hoveredWire, setHoveredWire] = useState<HoveredTooltip | null>(null);
  const [hoveredPin, setHoveredPin] = useState<HoveredTooltip | null>(null);

  const editor = useMemo(() => <MaterialGraphEditor document={document} graph={graph} />, [document, graph]);

  const pinMetadata = useMemo(() => {
    const metadata: MaterialPinMetadata[] = [];
    for (const node of graph.nodes) {
      const definition = materialGraphDomain.getNodeDefinition(node);
      for (const pin of definition.inputs) {
        const entry = { direction: 'input' as const, node, pin };
        metadata.push({ ...entry, key: graphPinKey(node.id, pin.id, false), tooltip: materialPinTooltip(entry) });
      }
      for (const pin of definition.outputs) {
        const entry = { direction: 'output' as const, node, pin };
        metadata.push({ ...entry, key: graphPinKey(node.id, pin.id, true), tooltip: materialPinTooltip(entry) });
      }
    }
    return metadata;
  }, [graph.nodes]);

  const pinMetadataByKey = useMemo(
    () => new Map(pinMetadata.map((metadata) => [metadata.key, metadata])),
    [pinMetadata],
  );

  const wireMetadata = useMemo(() => {
    return new Map(
      graph.connections.flatMap((connection) => {
        const fromNode = graph.nodes.find((node) => node.id === connection.from.nodeId);
        const toNode = graph.nodes.find((node) => node.id === connection.to.nodeId);
        if (!fromNode || !toNode) return [];
        const fromDefinition = materialGraphDomain.getNodeDefinition(fromNode);
        const toDefinition = materialGraphDomain.getNodeDefinition(toNode);
        const fromPin = fromDefinition.outputs.find((pin) => pin.id === connection.from.pin);
        const toPin = toDefinition.inputs.find((pin) => pin.id === connection.to.pin);
        if (!fromPin || !toPin) return [];
        return [
          [
            connection.id,
            {
              fromPinKey: graphPinKey(fromNode.id, fromPin.id, true),
              label: `${materialPinTypeLabel(fromPin.type)} • ${fromDefinition.title}.${fromPin.id} → ${toPin.label}`,
              toPinKey: graphPinKey(toNode.id, toPin.id, false),
            },
          ] as const,
        ];
      }),
    );
  }, [graph.connections, graph.nodes]);

  const measureWires = useCallback(() => {
    const host = hostRef.current;
    if (!host) return;
    const hostRect = host.getBoundingClientRect();
    const elements = graphPinElementMap(host);
    const next = graph.connections.flatMap((connection) => {
      const metadata = wireMetadata.get(connection.id);
      if (!metadata) return [];
      const fromElement = elements.get(metadata.fromPinKey);
      const toElement = elements.get(metadata.toPinKey);
      if (!fromElement || !toElement) return [];
      const from = pinSocketCenter(fromElement, hostRect);
      const to = pinSocketCenter(toElement, hostRect);
      if (!from || !to) return [];
      return [
        {
          fromPinKey: metadata.fromPinKey,
          id: connection.id,
          label: metadata.label,
          path: graphConnectionPath(from, to),
          toPinKey: metadata.toPinKey,
        },
      ];
    });
    setWires((current) => (sameWireOverlays(current, next) ? current : next));
  }, [graph.connections, wireMetadata]);

  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let frame = window.requestAnimationFrame(measureWires);
    measureWires();
    const scheduleMeasure = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(measureWires);
    };
    const observer =
      typeof ResizeObserver === 'undefined'
        ? null
        : new ResizeObserver(() => {
            scheduleMeasure();
          });
    observer?.observe(host);
    for (const node of host.querySelectorAll<HTMLElement>('.material-graph-node')) observer?.observe(node);
    host.addEventListener('pointermove', scheduleMeasure);
    window.addEventListener('resize', scheduleMeasure);
    return () => {
      window.cancelAnimationFrame(frame);
      observer?.disconnect();
      host.removeEventListener('pointermove', scheduleMeasure);
      window.removeEventListener('resize', scheduleMeasure);
    };
  }, [measureWires]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const elements = graphPinElementMap(host);
    const cleanups: Array<() => void> = [];
    for (const metadata of pinMetadata) {
      const element = elements.get(metadata.key);
      if (!element) continue;
      const show = (event: PointerEvent) => {
        setHoveredPin({
          id: metadata.key,
          position: localPointerPosition(host, event.clientX, event.clientY),
          text: metadata.tooltip,
        });
      };
      const hide = () => setHoveredPin((current) => (current?.id === metadata.key ? null : current));
      element.addEventListener('pointerenter', show);
      element.addEventListener('pointermove', show);
      element.addEventListener('pointerleave', hide);
      cleanups.push(() => {
        element.removeEventListener('pointerenter', show);
        element.removeEventListener('pointermove', show);
        element.removeEventListener('pointerleave', hide);
      });
    }
    return () => cleanups.forEach((cleanup) => cleanup());
  }, [pinMetadata]);

  const hoveredWireId = hoveredWire?.id;
  const flowWireIds = useMemo(
    () => (hoveredWireId ? materialConnectionFlowIds(graph, hoveredWireId) : new Set<string>()),
    [graph, hoveredWireId],
  );

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const primaryEndpointKeys = new Set<string>();
    const flowEndpointKeys = new Set<string>();
    for (const wire of wires) {
      if (!flowWireIds.has(wire.id)) continue;
      flowEndpointKeys.add(wire.fromPinKey);
      flowEndpointKeys.add(wire.toPinKey);
      if (wire.id === hoveredWireId) {
        primaryEndpointKeys.add(wire.fromPinKey);
        primaryEndpointKeys.add(wire.toPinKey);
      }
    }
    const elements = graphPinElementMap(host);
    for (const [key, element] of elements) {
      element.classList.toggle('is-wire-flow-endpoint', flowEndpointKeys.has(key));
      element.classList.toggle('is-wire-endpoint', primaryEndpointKeys.has(key));
    }
    return () => {
      for (const element of elements.values()) {
        element.classList.remove('is-wire-flow-endpoint', 'is-wire-endpoint');
      }
    };
  }, [flowWireIds, hoveredWireId, wires]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const clearConnectionTargets = () => {
      for (const element of host.querySelectorAll<HTMLElement>('.material-pin')) {
        element.classList.remove('is-connection-source', 'is-compatible-target', 'is-incompatible-target');
      }
    };

    const handlePointerDown = (event: PointerEvent) => {
      if (document.readOnly || event.button !== 0) return;
      const target = event.target;
      if (!(target instanceof Element)) return;
      const pinElement = target.closest<HTMLElement>('[data-graph-pin-key]');
      if (pinElement && host.contains(pinElement)) {
        const key = pinElement.dataset.graphPinKey;
        const source = key ? pinMetadataByKey.get(key) : undefined;
        if (!source) return;
        if (source.direction === 'input') {
          clearConnectionTargets();
          return;
        }

        clearConnectionTargets();
        pinElement.classList.add('is-connection-source');
        const elements = graphPinElementMap(host);
        for (const metadata of pinMetadata) {
          if (metadata.direction !== 'input') continue;
          const targetElement = elements.get(metadata.key);
          if (!targetElement) continue;
          const allowed = materialGraphDomain.canConnect(
            { node: source.node, pin: source.pin, direction: 'output' },
            { node: metadata.node, pin: metadata.pin, direction: 'input' },
          ).allowed;
          targetElement.classList.add(allowed ? 'is-compatible-target' : 'is-incompatible-target');
        }
        return;
      }

      const canvas = host.querySelector<HTMLElement>('.material-graph-canvas');
      if (target === canvas) clearConnectionTargets();
    };

    host.addEventListener('pointerdown', handlePointerDown, true);
    return () => {
      host.removeEventListener('pointerdown', handlePointerDown, true);
      clearConnectionTargets();
    };
  }, [document.readOnly, pinMetadata, pinMetadataByKey]);

  const tooltip = hoveredWire ?? hoveredPin;

  return (
    <div className="material-graph-interaction-host" ref={hostRef}>
      {editor}
      <svg aria-hidden="true" className="material-graph-interaction-overlay">
        {wires.map((wire) => {
          const isFlow = flowWireIds.has(wire.id);
          const isPrimary = wire.id === hoveredWireId;
          return (
            <g
              className={`material-wire-interaction${isFlow ? ' is-flow' : ''}${isPrimary ? ' is-primary' : ''}`}
              data-material-wire-id={wire.id}
              key={wire.id}
            >
              <path className="material-wire-flow-glow" d={wire.path} />
              <path className="material-wire-flow-texture" d={wire.path} pathLength={100} />
              <path
                className="material-wire-hit"
                d={wire.path}
                onPointerEnter={(event) =>
                  setHoveredWire({
                    id: wire.id,
                    position: localPointerPosition(hostRef.current!, event.clientX, event.clientY),
                    text: wire.label,
                  })
                }
                onPointerLeave={() => setHoveredWire((current) => (current?.id === wire.id ? null : current))}
                onPointerMove={(event) =>
                  setHoveredWire({
                    id: wire.id,
                    position: localPointerPosition(hostRef.current!, event.clientX, event.clientY),
                    text: wire.label,
                  })
                }
              />
            </g>
          );
        })}
      </svg>
      {tooltip && (
        <div
          className={`material-graph-hover-tooltip ${hoveredWire ? 'is-wire-tooltip' : 'is-pin-tooltip'}`}
          role="tooltip"
          style={{ left: tooltip.position[0] + 12, top: tooltip.position[1] + 14 }}
        >
          {tooltip.text}
        </div>
      )}
    </div>
  );
}
