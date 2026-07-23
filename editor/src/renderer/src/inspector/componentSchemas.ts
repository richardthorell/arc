import type { InspectorEntitySnapshot } from './inspectorTypes';
import type { PropertyComponentSchema, PropertyFieldSchema } from './propertySchema';
import { generatedEcsComponents } from './generatedEcsMetadata';
export { getPathValue, setPathValue } from './propertySchema';

export type InspectorComponentId = 'transform' | 'camera' | 'meshRenderer' | 'terrain' | 'prefab';
export type InspectorFieldSchema = PropertyFieldSchema<InspectorEntitySnapshot>;
export type InspectorComponentSchema = PropertyComponentSchema<InspectorEntitySnapshot, InspectorComponentId>;

const generatedTitle = (canonicalName: string, fallback: string) => (
  generatedEcsComponents.find((component) => component.canonicalName === canonicalName)?.displayName ?? fallback
);

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
      : schema.id === 'meshRenderer' ? snapshot.meshRenderer !== null
        : schema.id === 'prefab' ? snapshot.prefab !== null
          : snapshot.terrain !== null
));
