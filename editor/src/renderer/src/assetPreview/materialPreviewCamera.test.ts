import { describe, expect, it } from 'vitest';

import {
  clampMaterialPreviewOrbitY,
  clampMaterialPreviewZoom,
  constrainMaterialPreviewPitchToFloor,
  materialPreviewDefaultCameraDistance,
  materialPreviewInitialCameraPitch,
  materialPreviewInitialZoom,
  materialPreviewMaximumPitchForDistance,
  materialPreviewMinimumCameraDistance,
  materialPreviewNativeCameraDistance,
} from './materialPreviewCamera';

describe('material preview camera framing', () => {
  it('starts closer than the native fallback while remaining outside the sphere', () => {
    expect(materialPreviewDefaultCameraDistance).toBeLessThan(materialPreviewNativeCameraDistance);
    expect(materialPreviewDefaultCameraDistance).toBeGreaterThan(materialPreviewMinimumCameraDistance);

    const framed = clampMaterialPreviewZoom(materialPreviewNativeCameraDistance, materialPreviewInitialZoom);
    expect(framed.distance).toBeCloseTo(materialPreviewDefaultCameraDistance, 6);
  });

  it('stops zoom-in motion before the camera can enter the sphere', () => {
    const zoomed = clampMaterialPreviewZoom(0.7, 2);
    expect(zoomed.distance).toBe(materialPreviewMinimumCameraDistance);
    expect(zoomed.zoom).toBeGreaterThan(0);

    const blocked = clampMaterialPreviewZoom(materialPreviewMinimumCameraDistance, 2);
    expect(blocked).toEqual({ distance: materialPreviewMinimumCameraDistance, zoom: 0 });
  });

  it('keeps zoom-out motion unrestricted', () => {
    const zoomed = clampMaterialPreviewZoom(materialPreviewMinimumCameraDistance, -1);
    expect(zoomed.distance).toBeGreaterThan(materialPreviewMinimumCameraDistance);
    expect(zoomed.zoom).toBe(-1);
  });

  it('stops orbiting before the material camera can pass through the studio floor', () => {
    const clamped = clampMaterialPreviewOrbitY(materialPreviewInitialCameraPitch, 1.55, -200);
    expect(clamped.pitch).toBeCloseTo(materialPreviewMaximumPitchForDistance(1.55), 6);
    expect(Math.abs(clamped.orbitY)).toBeLessThan(200);

    const blocked = clampMaterialPreviewOrbitY(clamped.pitch, 1.55, -20);
    expect(blocked.pitch).toBeCloseTo(clamped.pitch, 6);
    expect(blocked.orbitY).toBeCloseTo(0, 6);
  });

  it('adjusts pitch when a distance change would otherwise put the camera below the floor', () => {
    const closePitch = materialPreviewMaximumPitchForDistance(0.7);
    const corrected = constrainMaterialPreviewPitchToFloor(closePitch, 2.0);
    expect(corrected.pitch).toBeCloseTo(materialPreviewMaximumPitchForDistance(2.0), 6);
    expect(corrected.orbitY).toBeGreaterThan(0);
  });
});
