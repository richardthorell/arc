import type { InspectorEntitySnapshot } from './inspectorTypes';
import type { PropertyComponentSchema, PropertyFieldSchema } from './propertySchema';
export { getPathValue, setPathValue } from './propertySchema';

export type InspectorComponentId = 'transform' | 'camera';
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
];

export const schemaForSnapshot = (snapshot: InspectorEntitySnapshot) => inspectorComponentSchemas.filter((schema) => (
  schema.id === 'transform' ? snapshot.transform !== null : snapshot.camera !== null
));
