import type { InspectorEntitySnapshot } from './inspectorTypes';
import type { PropertyComponentSchema, PropertyFieldSchema } from './propertySchema';
export { getPathValue, setPathValue } from './propertySchema';

export type InspectorComponentId = 'transform' | 'camera' | 'meshRenderer' | 'terrain' | 'terrainBrush';
export type InspectorFieldSchema = PropertyFieldSchema<InspectorEntitySnapshot>;
export type InspectorComponentSchema = PropertyComponentSchema<InspectorEntitySnapshot, InspectorComponentId>;

export const inspectorComponentSchemas: ReadonlyArray<InspectorComponentSchema> = [
  {
    id: 'transform',
    title: 'Transform',
    fields: [
      { id: 'position', label: 'Location', path: 'transform.position', type: 'vector3', precision: 2, step: 0.1, scrubSensitivity: 0.02 },
      { id: 'rotation', label: 'Rotation', path: 'transform.rotationDegrees', type: 'vector3', precision: 1, step: 1, scrubSensitivity: 0.2, unit: '°' },
      { id: 'scale', label: 'Scale', path: 'transform.scale', type: 'vector3', precision: 2, step: 0.01, scrubSensitivity: 0.01, linked: true },
    ],
  },
  {
    id: 'camera',
    title: 'Camera',
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
    ],
  },
  {
    id: 'meshRenderer',
    title: 'Mesh Renderer',
    fields: [
      { id: 'preview', label: 'Material Preview', path: 'meshRenderer.materialPath', namePath: 'meshRenderer.materialName', type: 'assetPreview', assetKind: 'material' },
      { id: 'material', label: 'Material', path: 'meshRenderer.materialPath', type: 'asset', assetKind: 'material', allowedExtensions: ['.arcmat'], allowEmpty: false },
      { id: 'visible', label: 'Visible', path: 'meshRenderer.visible', type: 'boolean' },
      { id: 'tint', label: 'Color Tint', path: 'meshRenderer.baseColorTint', type: 'color', precision: 2, min: 0, max: 1 },
    ],
  },
  {
    id: 'terrain',
    title: 'Terrain',
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
    id: 'terrainBrush',
    title: 'Terrain Brush',
    fields: [
      { id: 'tool', label: 'Tool', path: 'terrain.brushTool', type: 'enum', options: [
        { value: 'sculpt', label: 'Sculpt' }, { value: 'smooth', label: 'Smooth' },
        { value: 'flatten', label: 'Flatten' }, { value: 'paint', label: 'Paint' },
      ] },
      { id: 'radius', label: 'Radius', path: 'terrain.brushRadius', type: 'number', precision: 2, step: 0.5, scrubSensitivity: 0.05, unit: ' m', min: 0.25, max: 128 },
      { id: 'strength', label: 'Strength', path: 'terrain.brushStrength', type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0.001, max: 1 },
      { id: 'falloff', label: 'Smooth Falloff', path: 'terrain.brushFalloff', type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0, max: 1 },
      { id: 'activeLayer', label: 'Paint Layer', path: 'terrain.activeLayer', type: 'enum', options: [
        { value: '0', label: 'Grass' }, { value: '1', label: 'Dirt' },
        { value: '2', label: 'Rock' }, { value: '3', label: 'Sand' },
      ], visible: (snapshot) => snapshot.terrain?.brushTool === 'paint' },
    ],
  },
];

export const schemaForSnapshot = (snapshot: InspectorEntitySnapshot) => inspectorComponentSchemas.filter((schema) => (
  schema.id === 'transform' ? snapshot.transform !== null
    : schema.id === 'camera' ? snapshot.camera !== null
      : schema.id === 'meshRenderer' ? snapshot.meshRenderer !== null
        : snapshot.terrain !== null
));
