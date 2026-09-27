export type TerrainRegion = Readonly<{
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}>;

export type TerrainStrokeKind = 'sculpt' | 'paint';

export type TerrainStrokeRequest = Readonly<{
  terrainAsset: string;
  expectedRevision: string;
  layerId: string;
  kind: TerrainStrokeKind;
  region: TerrainRegion;
  strength: number;
  radius: number;
  channel?: string;
}>;

export type TerrainModifierRequest = Readonly<{
  terrainAsset: string;
  expectedRevision: string;
  modifierId: string;
  insertAfterId?: string;
}>;

export type TerrainOperationValidation = Readonly<{
  valid: boolean;
  diagnostics: readonly string[];
}>;

const assetPathPattern = /^(?![\\/])(?!.*(?:^|[\\/])\.\.(?:[\\/]|$)).+\.arcterrain$/i;
const revisionPattern = /^[a-f0-9]{64}$/i;
const stableIdPattern = /^[A-Za-z0-9][A-Za-z0-9_.:-]*$/;

function finite(value: number): boolean {
  return Number.isFinite(value);
}

function validateCommon(terrainAsset: string, expectedRevision: string): string[] {
  const diagnostics: string[] = [];
  if (!assetPathPattern.test(terrainAsset) || terrainAsset.includes('\\')) {
    diagnostics.push('terrainAsset must be a normalized project-relative .arcterrain path');
  }
  if (!revisionPattern.test(expectedRevision)) {
    diagnostics.push('expectedRevision must be a SHA-256 revision');
  }
  return diagnostics;
}

export function validateTerrainRegion(region: TerrainRegion): TerrainOperationValidation {
  const diagnostics: string[] = [];
  if (![region.minX, region.minY, region.maxX, region.maxY].every(finite)) {
    diagnostics.push('terrain region coordinates must be finite');
  } else if (region.minX >= region.maxX || region.minY >= region.maxY) {
    diagnostics.push('terrain region must have positive area');
  }
  return { valid: diagnostics.length === 0, diagnostics };
}

export function validateTerrainStroke(request: TerrainStrokeRequest): TerrainOperationValidation {
  const diagnostics = validateCommon(request.terrainAsset, request.expectedRevision);
  if (!stableIdPattern.test(request.layerId)) diagnostics.push('layerId must be a stable non-empty identifier');
  diagnostics.push(...validateTerrainRegion(request.region).diagnostics);
  if (!finite(request.strength) || request.strength < -1 || request.strength > 1 || request.strength === 0) {
    diagnostics.push('strength must be finite, non-zero, and within [-1, 1]');
  }
  if (!finite(request.radius) || request.radius <= 0) diagnostics.push('radius must be finite and greater than zero');
  if (request.kind === 'paint' && (!request.channel || !stableIdPattern.test(request.channel))) {
    diagnostics.push('paint strokes require a stable channel identifier');
  }
  if (request.kind === 'sculpt' && request.channel !== undefined) {
    diagnostics.push('sculpt strokes cannot specify a paint channel');
  }
  return { valid: diagnostics.length === 0, diagnostics };
}

export function validateTerrainModifier(request: TerrainModifierRequest): TerrainOperationValidation {
  const diagnostics = validateCommon(request.terrainAsset, request.expectedRevision);
  if (!stableIdPattern.test(request.modifierId)) diagnostics.push('modifierId must be a stable non-empty identifier');
  if (request.insertAfterId !== undefined && !stableIdPattern.test(request.insertAfterId)) {
    diagnostics.push('insertAfterId must be a stable identifier when supplied');
  }
  if (request.insertAfterId === request.modifierId) diagnostics.push('a modifier cannot be ordered after itself');
  return { valid: diagnostics.length === 0, diagnostics };
}
