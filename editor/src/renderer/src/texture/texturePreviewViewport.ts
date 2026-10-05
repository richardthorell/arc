export type TexturePreviewSize = {
  width: number;
  height: number;
};

export const TEXTURE_PREVIEW_MIN_ZOOM = 0.01;
export const TEXTURE_PREVIEW_MAX_ZOOM = 64;

const positiveFinite = (value: number) => Number.isFinite(value) && value > 0;

export const clampTexturePreviewZoom = (zoom: number) => {
  if (!Number.isFinite(zoom)) return 1;
  return Math.min(TEXTURE_PREVIEW_MAX_ZOOM, Math.max(TEXTURE_PREVIEW_MIN_ZOOM, zoom));
};

/**
 * Returns a deterministic zoom that fits the complete texture inside the available preview area.
 * Invalid/zero geometry falls back to 1x so transient resize or unloaded-image states cannot
 * poison persisted view state with NaN/Infinity.
 */
export const fitTexturePreviewZoom = (
  texture: TexturePreviewSize,
  viewport: TexturePreviewSize,
  padding = 0,
) => {
  if (
    !positiveFinite(texture.width) ||
    !positiveFinite(texture.height) ||
    !positiveFinite(viewport.width) ||
    !positiveFinite(viewport.height) ||
    !Number.isFinite(padding)
  )
    return 1;

  const inset = Math.max(0, padding) * 2;
  const availableWidth = viewport.width - inset;
  const availableHeight = viewport.height - inset;
  if (!positiveFinite(availableWidth) || !positiveFinite(availableHeight)) return TEXTURE_PREVIEW_MIN_ZOOM;

  return clampTexturePreviewZoom(
    Math.min(availableWidth / texture.width, availableHeight / texture.height),
  );
};

export const zoomTexturePreviewAroundPoint = (
  previousZoom: number,
  nextZoom: number,
  pan: { x: number; y: number },
  anchor: { x: number; y: number },
) => {
  const from = clampTexturePreviewZoom(previousZoom);
  const to = clampTexturePreviewZoom(nextZoom);
  const ratio = to / from;
  return {
    zoom: to,
    pan: {
      x: anchor.x - (anchor.x - pan.x) * ratio,
      y: anchor.y - (anchor.y - pan.y) * ratio,
    },
  };
};
