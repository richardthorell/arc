export type TexturePreviewSize = { width: number; height: number };
export type TexturePreviewPoint = { x: number; y: number };

export const TEXTURE_PREVIEW_MIN_ZOOM = 0.05;
export const TEXTURE_PREVIEW_MAX_ZOOM = 32;

export const clampTexturePreviewZoom = (zoom: number) =>
  Math.min(TEXTURE_PREVIEW_MAX_ZOOM, Math.max(TEXTURE_PREVIEW_MIN_ZOOM, zoom));

export const fitTexturePreview = (texture: TexturePreviewSize, viewport: TexturePreviewSize, padding = 24) => {
  if (texture.width <= 0 || texture.height <= 0 || viewport.width <= 0 || viewport.height <= 0) {
    return { zoom: 1, pan: { x: 0, y: 0 } satisfies TexturePreviewPoint };
  }

  const availableWidth = Math.max(1, viewport.width - padding * 2);
  const availableHeight = Math.max(1, viewport.height - padding * 2);
  const zoom = clampTexturePreviewZoom(Math.min(availableWidth / texture.width, availableHeight / texture.height));

  return { zoom, pan: { x: 0, y: 0 } satisfies TexturePreviewPoint };
};

export const zoomTexturePreviewAroundPoint = (
  currentZoom: number,
  nextZoom: number,
  pan: TexturePreviewPoint,
  anchor: TexturePreviewPoint,
): { zoom: number; pan: TexturePreviewPoint } => {
  const zoom = clampTexturePreviewZoom(nextZoom);
  const safeCurrentZoom = clampTexturePreviewZoom(currentZoom);
  const scale = zoom / safeCurrentZoom;

  return {
    zoom,
    pan: {
      x: anchor.x - (anchor.x - pan.x) * scale,
      y: anchor.y - (anchor.y - pan.y) * scale,
    },
  };
};
