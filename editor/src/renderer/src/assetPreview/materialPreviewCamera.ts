export const materialPreviewSphereRadius = 0.5;
export const materialPreviewSurfaceClearance = 0.06;
export const materialPreviewMinimumCameraDistance = materialPreviewSphereRadius + materialPreviewSurfaceClearance;
export const materialPreviewDefaultCameraDistance = 1.55;
export const materialPreviewFloorSurfaceY = -0.5;
export const materialPreviewCameraFloorClearance = 0.08;
export const materialPreviewMinimumCameraY = materialPreviewFloorSurfaceY + materialPreviewCameraFloorClearance;

// The native preview currently starts at { 1.65, 0.55, 2.25 } looking at the origin.
// Keep these values centralized until preview-camera framing becomes a host-level setting.
export const materialPreviewNativeCameraDistance = Math.hypot(1.65, 0.55, 2.25);
export const materialPreviewInitialCameraPitch = Math.asin(-0.55 / materialPreviewNativeCameraDistance);
export const materialPreviewCameraDollyUnits = 1.5;
export const materialPreviewCameraOrbitRadiansPerPixel = 0.008;
export const materialPreviewCameraMaximumPitch = 1.45;
export const materialPreviewInitialZoom =
  (materialPreviewNativeCameraDistance - materialPreviewDefaultCameraDistance) / materialPreviewCameraDollyUnits;

export type MaterialPreviewZoom = {
  distance: number;
  zoom: number;
};

export type MaterialPreviewOrbit = {
  orbitY: number;
  pitch: number;
};

const validMaterialPreviewDistance = (distance: number) =>
  Number.isFinite(distance) && distance > 0 ? distance : materialPreviewDefaultCameraDistance;

/** Return the steepest downward-looking orbit pitch that still keeps the camera above the studio floor. */
export function materialPreviewMaximumPitchForDistance(distance: number): number {
  const currentDistance = validMaterialPreviewDistance(distance);
  const floorRatio = Math.min(1, Math.max(-1, -materialPreviewMinimumCameraY / currentDistance));
  return Math.min(materialPreviewCameraMaximumPitch, Math.asin(floorRatio));
}

/** Clamp material-preview orbit motion so the camera center cannot rotate below the studio floor. */
export function clampMaterialPreviewOrbitY(
  currentPitch: number,
  currentDistance: number,
  requestedOrbitY: number,
): MaterialPreviewOrbit {
  const pitch = Number.isFinite(currentPitch) ? currentPitch : materialPreviewInitialCameraPitch;
  if (!Number.isFinite(requestedOrbitY) || requestedOrbitY === 0) return { orbitY: 0, pitch };

  const requestedPitch = Math.max(
    -materialPreviewCameraMaximumPitch,
    Math.min(materialPreviewCameraMaximumPitch, pitch - requestedOrbitY * materialPreviewCameraOrbitRadiansPerPixel),
  );
  const nextPitch = Math.min(requestedPitch, materialPreviewMaximumPitchForDistance(currentDistance));
  return {
    orbitY: (pitch - nextPitch) / materialPreviewCameraOrbitRadiansPerPixel,
    pitch: nextPitch,
  };
}

/** Keep the current pitch valid after dolly distance changes, returning the orbit correction needed by the host. */
export function constrainMaterialPreviewPitchToFloor(currentPitch: number, currentDistance: number): MaterialPreviewOrbit {
  const pitch = Number.isFinite(currentPitch) ? currentPitch : materialPreviewInitialCameraPitch;
  const nextPitch = Math.min(pitch, materialPreviewMaximumPitchForDistance(currentDistance));
  return {
    orbitY: (pitch - nextPitch) / materialPreviewCameraOrbitRadiansPerPixel,
    pitch: nextPitch,
  };
}

/** Clamp material-preview dolly motion so the camera center never enters the preview sphere. */
export function clampMaterialPreviewZoom(currentDistance: number, requestedZoom: number): MaterialPreviewZoom {
  const distance = validMaterialPreviewDistance(currentDistance);
  if (!Number.isFinite(requestedZoom) || requestedZoom === 0) return { distance, zoom: 0 };

  const requestedDistance = distance - requestedZoom * materialPreviewCameraDollyUnits;
  const nextDistance =
    requestedZoom > 0 ? Math.max(materialPreviewMinimumCameraDistance, requestedDistance) : requestedDistance;
  return {
    distance: nextDistance,
    zoom: (distance - nextDistance) / materialPreviewCameraDollyUnits,
  };
}
