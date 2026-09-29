export type TextureViewportSize = {
  width: number;
  height: number;
};

export type TextureFitZoomOptions = {
  viewport: TextureViewportSize;
  texture: TextureViewportSize;
  padding: number;
  minZoom: number;
  maxZoom: number;
};

const isPositiveFinite = (value: number) => Number.isFinite(value) && value > 0;

export const calculateTextureFitZoom = ({
  viewport,
  texture,
  padding,
  minZoom,
  maxZoom,
}: TextureFitZoomOptions): number | null => {
  if (
    !isPositiveFinite(viewport.width) ||
    !isPositiveFinite(viewport.height) ||
    !isPositiveFinite(texture.width) ||
    !isPositiveFinite(texture.height) ||
    !Number.isFinite(padding) ||
    padding < 0 ||
    !isPositiveFinite(minZoom) ||
    !isPositiveFinite(maxZoom) ||
    minZoom > maxZoom
  ) {
    return null;
  }

  const availableWidth = Math.max(1, viewport.width - padding * 2);
  const availableHeight = Math.max(1, viewport.height - padding * 2);
  const fit = Math.min(availableWidth / texture.width, availableHeight / texture.height);
  return Math.min(maxZoom, Math.max(minZoom, fit));
};
