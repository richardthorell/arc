const clamp = (value: number, minimum: number, maximum: number) => Math.min(maximum, Math.max(minimum, value));

/**
 * Normalize browser wheel deltas into the signed wheel-step units used by the
 * native editor camera. Positive values zoom in and negative values zoom out.
 *
 * Chromium reports traditional mouse wheels as fairly large pixel deltas while
 * high-resolution wheels and trackpads can report much smaller values. Keep
 * those smaller gestures useful without letting a single event produce an
 * extreme camera jump.
 */
export function normalizeViewportWheel(deltaY: number, deltaMode = 0): number {
  if (!Number.isFinite(deltaY) || deltaY === 0) return 0;

  const direction = deltaY < 0 ? 1 : -1;
  const magnitude = Math.abs(deltaY);
  let steps: number;

  if (deltaMode === 1) {
    // DOM_DELTA_LINE: most wheel devices report one or a few lines per notch.
    steps = magnitude;
  } else if (deltaMode === 2) {
    // DOM_DELTA_PAGE is rare for camera input; treat it as a deliberate larger step.
    steps = 4;
  } else {
    // DOM_DELTA_PIXEL: ~100-120 px is a common mouse-wheel notch in Chromium.
    // Preserve smooth high-resolution input, but keep very small events visible.
    steps = magnitude / 100;
  }

  return direction * clamp(steps, 0.2, 4);
}
