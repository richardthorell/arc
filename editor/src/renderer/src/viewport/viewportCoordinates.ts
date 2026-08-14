export type ViewportCoordinateInput = {
  clientX: number;
  clientY: number;
  left: number;
  top: number;
  cssWidth: number;
  cssHeight: number;
  renderWidth: number;
  renderHeight: number;
  devicePixelRatio: number;
};

export const toViewportPixels = (input: ViewportCoordinateInput): { x: number; y: number } => {
  const scaleX = input.renderWidth > 0 ? input.renderWidth / Math.max(1, input.cssWidth) : input.devicePixelRatio;
  const scaleY = input.renderHeight > 0 ? input.renderHeight / Math.max(1, input.cssHeight) : input.devicePixelRatio;
  return {
    x: Math.max(0, Math.min(Math.max(0, input.renderWidth - 1), Math.round((input.clientX - input.left) * scaleX))),
    y: Math.max(0, Math.min(Math.max(0, input.renderHeight - 1), Math.round((input.clientY - input.top) * scaleY))),
  };
};
