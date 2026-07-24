import type { InspectorEntitySnapshot } from './inspectorTypes';
import type { PropertyComponentSchema, PropertyFieldSchema } from './propertySchema';
import { generatedEcsComponents } from './generatedEcsMetadata';
export { getPathValue, setPathValue } from './propertySchema';

export type InspectorComponentId = 'transform' | 'camera' | 'directionalLight' | 'pointLight' | 'spotLight' |
  'areaLight' | 'meshRenderer' | 'terrain' | 'prefab';
export type InspectorFieldSchema = PropertyFieldSchema<InspectorEntitySnapshot>;
export type InspectorComponentSchema = PropertyComponentSchema<InspectorEntitySnapshot, InspectorComponentId>;

const generatedTitle = (canonicalName: string, fallback: string) => (
  generatedEcsComponents.find((component) => component.canonicalName === canonicalName)?.displayName ?? fallback
);

const legacyUnit = { value: 'unitless', label: 'Legacy Unitless' };
const commonLightFieldsFor = (
  unitOptions: Array<{ value: string; label: string }>,
): InspectorFieldSchema[] => [
  { id: 'enabled', label: 'Enabled', path: 'light.enabled', type: 'boolean' },
  { id: 'color', label: 'Color', path: 'light.color', type: 'color', precision: 2, min: 0, max: 1 },
  { id: 'temperatureEnabled', label: 'Use Temperature', path: 'light.useColorTemperature', type: 'boolean' },
  {
    id: 'temperature', label: 'Temperature', path: 'light.temperatureKelvin', type: 'number',
    precision: 0, step: 50, scrubSensitivity: 5, min: 1000, max: 40000, unit: ' K',
    visible: (snapshot) => snapshot.light?.useColorTemperature ?? false,
  },
  { id: 'unit', label: 'Intensity Unit', path: 'light.unit', type: 'enum', options: [...unitOptions, legacyUnit] },
  { id: 'intensity', label: 'Intensity', path: 'light.intensity', type: 'number', precision: 1, step: 10, scrubSensitivity: 1, min: 0 },
  { id: 'shadows', label: 'Cast Shadows', path: 'light.castsShadows', type: 'boolean' },
];

export const inspectorComponentSchemas: ReadonlyArray<InspectorComponentSchema> = [
  {
    id: 'transform',
    title: generatedTitle('arc::scene.transform_component', 'Transform'),
    fields: [
      { id: 'position', label: 'Location', path: 'transform.position', type: 'vector3', precision: 2, step: 0.1, scrubSensitivity: 0.02 },
      { id: 'rotation', label: 'Rotation', path: 'transform.rotationDegrees', type: 'vector3', precision: 1, step: 1, scrubSensitivity: 0.2, unit: '°' },
      { id: 'scale', label: 'Scale', path: 'transform.scale', type: 'vector3', precision: 2, step: 0.01, scrubSensitivity: 0.01, linked: true },
    ],
  },
  {
    id: 'camera',
    title: generatedTitle('arc::scene.camera_component', 'Camera'),
    fields: [
      {
        id: 'projection', label: 'Projection', path: 'camera.projection', type: 'enum',
        options: [{ value: 'perspective', label: 'Perspective' }, { value: 'orthographic', label: 'Orthographic' }],
      },
      {
        id: 'fov', label: 'Field of View', path: 'camera.fovYDegrees', type: 'number', precision: 1,
        step: 1, scrubSensitivity: 0.2, unit: '°', min: 1.01, max: 178.99,
        visible: (snapshot) => snapshot.camera?.projection === 'perspective',
      },
      {
        id: 'orthographicHeight', label: 'Ortho Size', path: 'camera.orthographicHeight', type: 'number', precision: 2,
        step: 0.1, scrubSensitivity: 0.02, min: 0.001,
        visible: (snapshot) => snapshot.camera?.projection === 'orthographic',
      },
      { id: 'nearPlane', label: 'Near Clip', path: 'camera.nearPlane', type: 'number', precision: 3, step: 0.01, scrubSensitivity: 0.002, min: 0.001 },
      { id: 'farPlane', label: 'Far Clip', path: 'camera.farPlane', type: 'number', precision: 1, step: 10, scrubSensitivity: 1, min: 0.002 },
      { id: 'active', label: 'Active Camera', path: 'camera.active', type: 'boolean' },
      { id: 'clearColor', label: 'Clear Color', path: 'camera.clearColor', type: 'color', precision: 2, min: 0, max: 1 },
      {
        id: 'exposureMode', label: 'Exposure', path: 'camera.exposureMode', type: 'enum',
        options: [{ value: 'automatic', label: 'Automatic' }, { value: 'manual', label: 'Manual' }],
      },
      {
        id: 'metering', label: 'Metering', path: 'camera.exposureMetering', type: 'enum',
        options: [{ value: 'average', label: 'Average' }, { value: 'centerWeighted', label: 'Center Weighted' }],
        visible: (snapshot) => snapshot.camera?.exposureMode === 'automatic',
      },
      {
        id: 'manualEV100', label: 'EV100', path: 'camera.manualEV100', type: 'number',
        precision: 2, step: 0.25, scrubSensitivity: 0.05, unit: ' EV',
        visible: (snapshot) => snapshot.camera?.exposureMode === 'manual',
      },
      {
        id: 'exposureCompensation', label: 'Compensation', path: 'camera.exposureCompensation', type: 'number',
        precision: 2, step: 0.25, scrubSensitivity: 0.05, unit: ' EV',
      },
      {
        id: 'minimumEV100', label: 'Minimum EV', path: 'camera.minimumEV100', type: 'number',
        precision: 1, step: 0.5, scrubSensitivity: 0.1,
        visible: (snapshot) => snapshot.camera?.exposureMode === 'automatic',
      },
      {
        id: 'maximumEV100', label: 'Maximum EV', path: 'camera.maximumEV100', type: 'number',
        precision: 1, step: 0.5, scrubSensitivity: 0.1,
        visible: (snapshot) => snapshot.camera?.exposureMode === 'automatic',
      },
      {
        id: 'brightenSpeed', label: 'Brighten Speed', path: 'camera.brightenSpeed', type: 'number',
        precision: 2, step: 0.1, scrubSensitivity: 0.02, min: 0,
        visible: (snapshot) => snapshot.camera?.exposureMode === 'automatic',
      },
      {
        id: 'darkenSpeed', label: 'Darken Speed', path: 'camera.darkenSpeed', type: 'number',
        precision: 2, step: 0.1, scrubSensitivity: 0.02, min: 0,
        visible: (snapshot) => snapshot.camera?.exposureMode === 'automatic',
      },
    ],
  },
  {
    id: 'directionalLight',
    title: generatedTitle('arc::scene.directional_light_component', 'Directional Light'),
    fields: commonLightFieldsFor([{ value: 'lux', label: 'Lux (lx)' }]),
  },
  {
    id: 'pointLight',
    title: generatedTitle('arc::scene.point_light_component', 'Point Light'),
    fields: [
      ...commonLightFieldsFor([
        { value: 'lumens', label: 'Lumens (lm)' },
        { value: 'candela', label: 'Candela (cd)' },
      ]),
      { id: 'range', label: 'Range', path: 'light.range', type: 'number', precision: 2, step: 0.25, scrubSensitivity: 0.05, min: 0.001, unit: ' m' },
    ],
  },
  {
    id: 'spotLight',
    title: generatedTitle('arc::scene.spot_light_component', 'Spot Light'),
    fields: [
      ...commonLightFieldsFor([
        { value: 'lumens', label: 'Lumens (lm)' },
        { value: 'candela', label: 'Candela (cd)' },
      ]),
      { id: 'range', label: 'Range', path: 'light.range', type: 'number', precision: 2, step: 0.25, scrubSensitivity: 0.05, min: 0.001, unit: ' m' },
      { id: 'innerAngle', label: 'Inner Angle', path: 'light.innerAngleDegrees', type: 'number', precision: 1, step: 1, scrubSensitivity: 0.2, min: 0, max: 178, unit: '°' },
      { id: 'outerAngle', label: 'Outer Angle', path: 'light.outerAngleDegrees', type: 'number', precision: 1, step: 1, scrubSensitivity: 0.2, min: 0.1, max: 178.9, unit: '°' },
    ],
  },
  {
    id: 'areaLight',
    title: generatedTitle('arc::scene.area_light_component', 'Area Light'),
    fields: [
      {
        id: 'shape', label: 'Shape', path: 'light.kind', type: 'enum',
        options: [{ value: 'rectangle', label: 'Rectangle' }, { value: 'disk', label: 'Disk' }],
      },
      ...commonLightFieldsFor([
        { value: 'lumens', label: 'Lumens (lm)' },
        { value: 'nits', label: 'Nits (cd/m²)' },
      ]),
      { id: 'width', label: 'Width', path: 'light.width', type: 'number', precision: 2, step: 0.1, scrubSensitivity: 0.02, min: 0.001, unit: ' m' },
      {
        id: 'height', label: 'Height', path: 'light.height', type: 'number', precision: 2,
        step: 0.1, scrubSensitivity: 0.02, min: 0.001, unit: ' m',
        visible: (snapshot) => snapshot.light?.kind === 'rectangle',
      },
      { id: 'twoSided', label: 'Two Sided', path: 'light.twoSided', type: 'boolean' },
    ],
  },
  {
    id: 'meshRenderer',
    title: generatedTitle('arc::scene.mesh_renderer_component', 'Mesh Renderer'),
    fields: [
      { id: 'preview', label: 'Material Preview', path: 'meshRenderer.materialPath', namePath: 'meshRenderer.materialName', type: 'assetPreview', assetKind: 'material' },
      { id: 'material', label: 'Material', path: 'meshRenderer.materialPath', type: 'asset', assetKind: 'material', allowedExtensions: ['.arcmat'], allowEmpty: false },
      { id: 'visible', label: 'Visible', path: 'meshRenderer.visible', type: 'boolean' },
      { id: 'tint', label: 'Color Tint', path: 'meshRenderer.baseColorTint', type: 'color', precision: 2, min: 0, max: 1 },
    ],
  },
  {
    id: 'terrain',
    title: generatedTitle('arc::scene.terrain_component', 'Terrain'),
    fields: [
      { id: 'enabled', label: 'Enabled', path: 'terrain.enabled', type: 'boolean' },
      { id: 'receiveShadows', label: 'Receive Shadows', path: 'terrain.receiveShadows', type: 'boolean' },
      { id: 'grass', label: 'Grass Layer', path: 'terrain.layers.0.baseColorPath', type: 'asset', assetKind: 'texture', allowedExtensions: ['.png', '.jpg', '.jpeg', '.tga'], allowEmpty: true },
      { id: 'dirt', label: 'Dirt Layer', path: 'terrain.layers.1.baseColorPath', type: 'asset', assetKind: 'texture', allowedExtensions: ['.png', '.jpg', '.jpeg', '.tga'], allowEmpty: true },
      { id: 'rock', label: 'Rock Layer', path: 'terrain.layers.2.baseColorPath', type: 'asset', assetKind: 'texture', allowedExtensions: ['.png', '.jpg', '.jpeg', '.tga'], allowEmpty: true },
      { id: 'sand', label: 'Sand Layer', path: 'terrain.layers.3.baseColorPath', type: 'asset', assetKind: 'texture', allowedExtensions: ['.png', '.jpg', '.jpeg', '.tga'], allowEmpty: true },
    ],
  },
  {
    id: 'prefab',
    title: 'Prefab Instance',
    badge: 'Instance',
    fields: [
      { id: 'source', label: 'Source', path: 'prefab.prefabPath', type: 'readonly' },
      { id: 'overrides', label: 'Overrides', path: 'prefab.overrideCount', type: 'readonly',
        format: (value) => `${Number(value)} authored override${Number(value) === 1 ? '' : 's'}` },
      { id: 'status', label: 'Source Status', path: 'prefab.sourceMissing', type: 'readonly',
        format: (value) => value ? 'Missing source' : 'Connected' },
      { id: 'actions', label: 'Instance', path: 'prefab.prefabGuid', type: 'actions', actions: [
        { id: 'apply', label: 'Apply', disabled: (snapshot) => snapshot.prefab?.sourceMissing ?? true },
        { id: 'revert', label: 'Revert', disabled: (snapshot) => snapshot.prefab?.sourceMissing ?? true },
        { id: 'unpack', label: 'Unpack', danger: true },
      ] },
    ],
  },
];

export const schemaForSnapshot = (snapshot: InspectorEntitySnapshot) => inspectorComponentSchemas.filter((schema) => (
  schema.id === 'transform' ? snapshot.transform !== null
    : schema.id === 'camera' ? snapshot.camera !== null
      : schema.id === 'directionalLight' ? snapshot.light?.kind === 'directional'
        : schema.id === 'pointLight' ? snapshot.light?.kind === 'point'
          : schema.id === 'spotLight' ? snapshot.light?.kind === 'spot'
            : schema.id === 'areaLight' ? snapshot.light?.kind === 'rectangle' || snapshot.light?.kind === 'disk'
      : schema.id === 'meshRenderer' ? snapshot.meshRenderer !== null
        : schema.id === 'prefab' ? snapshot.prefab !== null
          : snapshot.terrain !== null
));
