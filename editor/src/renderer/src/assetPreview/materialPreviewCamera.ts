export const materialPreviewSphereRadius = 0.5;
// The pill reaches 1.0 unit from the origin vertically and is the largest of
// the initial preview meshes. Keep one conservative boundary for all three so
// switching mesh cannot leave the camera inside the replacement primitive.
export const materialPreviewMaximumMeshExtent = 1.0;
export const materialPreviewSurfaceClearance = 0.2;
export const materialPreviewMinimumCameraDistance = materialPreviewMaximumMeshExtent + materialPreviewSurfaceClearance;
export const materialPreviewDefaultCameraDistance = 1.55;

// The native preview currently starts at { 1.65, 0.55, 2.25 } looking at the origin.
// Keep these values centralized until preview-camera framing becomes a host-level setting.
export const materialPreviewNativeCameraDistance = Math.hypot(1.65, 0.55, 2.25);
export const materialPreviewInitialCameraPitch = Math.asin(-0.55 / materialPreviewNativeCameraDistance);
export const materialPreviewCameraDollyUnits = 1.5;
export const materialPreviewCameraOrbitRadiansPerPixel = 0.008;
// Keep only the normal orbit pole guard. There is no floor in the HDRI preview,
// so vertical orbiting must not be constrained by a synthetic floor plane.
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

/** Clamp material-preview orbit only at the camera poles so the orbit cannot flip. */
export function clampMaterialPreviewOrbitY(currentPitch: number, requestedOrbitY: number): MaterialPreviewOrbit {
  const pitch = Number.isFinite(currentPitch) ? currentPitch : materialPreviewInitialCameraPitch;
  if (!Number.isFinite(requestedOrbitY) || requestedOrbitY === 0) return { orbitY: 0, pitch };

  const nextPitch = Math.max(
    -materialPreviewCameraMaximumPitch,
    Math.min(materialPreviewCameraMaximumPitch, pitch - requestedOrbitY * materialPreviewCameraOrbitRadiansPerPixel),
  );
  return {
    orbitY: (pitch - nextPitch) / materialPreviewCameraOrbitRadiansPerPixel,
    pitch: nextPitch,
  };
}

/** Clamp material-preview dolly motion so the camera keeps a useful gap from the preview mesh. */
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
