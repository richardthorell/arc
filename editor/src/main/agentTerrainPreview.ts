import {
  type TerrainModifierRequest,
  type TerrainRegion,
  type TerrainStrokeRequest,
  validateTerrainModifier,
  validateTerrainStroke,
} from './agentTerrainOperations';

export type TerrainEditRisk = 'bounded' | 'broad';

export type TerrainStrokePreview = Readonly<{
  kind: 'stroke';
  terrainAsset: string;
  expectedRevision: string;
  layerId: string;
  affectedRegion: TerrainRegion;
  affectedArea: number;
  risk: TerrainEditRisk;
  requiresApproval: boolean;
  summary: string;
}>;

export type TerrainModifierPreview = Readonly<{
  kind: 'modifier-order';
  terrainAsset: string;
  expectedRevision: string;
  modifierId: string;
  insertAfterId?: string;
  risk: 'bounded';
  requiresApproval: false;
  summary: string;
}>;

function regionArea(region: TerrainRegion): number {
  return (region.maxX - region.minX) * (region.maxY - region.minY);
}

/**
 * Builds an approval payload from an already asset-owned terrain request without mutating terrain state.
 * The caller supplies the broad-edit threshold so product policy stays outside this transport-neutral model.
 */
export function previewTerrainStroke(request: TerrainStrokeRequest, broadEditArea: number): TerrainStrokePreview {
  const validation = validateTerrainStroke(request);
  if (!validation.valid) throw new Error(validation.diagnostics.join('; '));
  if (!Number.isFinite(broadEditArea) || broadEditArea <= 0) {
    throw new Error('broadEditArea must be finite and greater than zero');
  }

  const affectedArea = regionArea(request.region);
  const broad = affectedArea > broadEditArea;
  return {
    kind: 'stroke',
    terrainAsset: request.terrainAsset,
    expectedRevision: request.expectedRevision,
    layerId: request.layerId,
    affectedRegion: { ...request.region },
    affectedArea,
    risk: broad ? 'broad' : 'bounded',
    requiresApproval: broad,
    summary: `${request.kind} ${request.layerId} over ${affectedArea} terrain units²`,
  };
}

/** Builds a non-mutating preview for a revision-owned modifier reorder. */
export function previewTerrainModifier(request: TerrainModifierRequest): TerrainModifierPreview {
  const validation = validateTerrainModifier(request);
  if (!validation.valid) throw new Error(validation.diagnostics.join('; '));

  return {
    kind: 'modifier-order',
    terrainAsset: request.terrainAsset,
    expectedRevision: request.expectedRevision,
    modifierId: request.modifierId,
    ...(request.insertAfterId ? { insertAfterId: request.insertAfterId } : {}),
    risk: 'bounded',
    requiresApproval: false,
    summary: request.insertAfterId
      ? `move modifier ${request.modifierId} after ${request.insertAfterId}`
      : `move modifier ${request.modifierId} to the start`,
  };
}
