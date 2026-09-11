import type { GraphPoint, GraphViewport } from './graphTypes';
import { sameGraphPointMaps } from './graphGeometry';

export const measureGraphPinPositions = (canvas: HTMLElement, viewport: GraphViewport) => {
  const canvasRect = canvas.getBoundingClientRect();
  const positions = new Map<string, GraphPoint>();
  for (const element of canvas.querySelectorAll<HTMLElement>('[data-graph-pin-key]')) {
    const key = element.dataset.graphPinKey;
    const socket = element.querySelector<HTMLElement>('[data-graph-pin-socket]');
    if (!key || !socket) continue;
    const rect = socket.getBoundingClientRect();
    positions.set(key, [
      (rect.left + rect.width / 2 - canvasRect.left - viewport.x) / viewport.zoom,
      (rect.top + rect.height / 2 - canvasRect.top - viewport.y) / viewport.zoom,
    ]);
  }
  return positions;
};

export const graphPinPositionsChanged = (current: Map<string, GraphPoint>, next: Map<string, GraphPoint>) =>
  !sameGraphPointMaps(current, next);
