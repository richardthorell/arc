import { describe, expect, it } from 'vitest';

import {
  clampMaterialPreviewOrbitY,
  clampMaterialPreviewZoom,
  materialPreviewCameraMaximumPitch,
  materialPreviewDefaultCameraDistance,
  materialPreviewInitialCameraPitch,
  materialPreviewInitialZoom,
  materialPreviewMaximumMeshExtent,
  materialPreviewMinimumCameraDistance,
  materialPreviewNativeCameraDistance,
  materialPreviewSphereRadius,
  materialPreviewSurfaceClearance,
} from './materialPreviewCamera';

describe('material preview camera framing', () => {
  it('starts closer than the native fallback while remaining outside the preview mesh', () => {
    expect(materialPreviewDefaultCameraDistance).toBeLessThan(materialPreviewNativeCameraDistance);
    expect(materialPreviewDefaultCameraDistance).toBeGreaterThan(materialPreviewMinimumCameraDistance);

    const framed = clampMaterialPreviewZoom(materialPreviewNativeCameraDistance, materialPreviewInitialZoom);
    expect(framed.distance).toBeCloseTo(materialPreviewDefaultCameraDistance, 6);
  });

  it('keeps a visible zoom boundary outside the sphere', () => {
    expect(materialPreviewMaximumMeshExtent).toBeGreaterThan(materialPreviewSphereRadius);
    expect(materialPreviewSurfaceClearance).toBeGreaterThan(0);
    expect(materialPreviewMinimumCameraDistance).toBeCloseTo(
      materialPreviewMaximumMeshExtent + materialPreviewSurfaceClearance,
      6,
    );

    const zoomed = clampMaterialPreviewZoom(1.0, 2);
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

  it('allows vertical orbiting past the old studio-floor limit', () => {
    const downward = clampMaterialPreviewOrbitY(materialPreviewInitialCameraPitch, -100);
    expect(downward.pitch).toBeGreaterThan(0);
    expect(downward.orbitY).toBeCloseTo(-100, 6);
  });

  it('only stops vertical orbit at the pole guard', () => {
    const clamped = clampMaterialPreviewOrbitY(materialPreviewInitialCameraPitch, -1000);
    expect(clamped.pitch).toBeCloseTo(materialPreviewCameraMaximumPitch, 6);
    expect(Math.abs(clamped.orbitY)).toBeLessThan(1000);

    const blocked = clampMaterialPreviewOrbitY(clamped.pitch, -20);
    expect(blocked.pitch).toBeCloseTo(clamped.pitch, 6);
    expect(blocked.orbitY).toBeCloseTo(0, 6);
  });
});
