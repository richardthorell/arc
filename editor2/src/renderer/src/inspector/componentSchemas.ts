import type { InspectorEntitySnapshot, Vec3, Vec4 } from './inspectorTypes';

export type InspectorComponentId = 'transform' | 'camera';
export type VectorAxis = keyof Vec3;
export type ColorChannel = keyof Vec4;

type FieldBase = {
  id: string;
  label: string;
  path: string;
  visible?: (snapshot: InspectorEntitySnapshot) => boolean;
};

export type Vector3FieldSchema = FieldBase & {
  type: 'vector3';
  precision: number;
  step: number;
  scrubSensitivity: number;
  unit?: string;
  linked?: boolean;
};

export type NumberFieldSchema = FieldBase & {
  type: 'number';
  precision: number;
  step: number;
  scrubSensitivity: number;
  unit?: string;
  min?: number;
  max?: number;
};

export type BooleanFieldSchema = FieldBase & { type: 'boolean' };
export type EnumFieldSchema = FieldBase & {
  type: 'enum';
  options: ReadonlyArray<{ value: string; label: string }>;
};
export type ColorFieldSchema = FieldBase & {
  type: 'color';
  precision: number;
  min: number;
  max: number;
};

export type InspectorFieldSchema =
  | Vector3FieldSchema
  | NumberFieldSchema
  | BooleanFieldSchema
  | EnumFieldSchema
  | ColorFieldSchema;

export type InspectorComponentSchema = {
  id: InspectorComponentId;
  title: string;
  fields: ReadonlyArray<InspectorFieldSchema>;
};

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

export function getPathValue(snapshot: InspectorEntitySnapshot, path: string): unknown {
  return path.split('.').reduce<unknown>((value, key) => (
    value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : undefined
  ), snapshot);
}

export function setPathValue(snapshot: InspectorEntitySnapshot, path: string, value: unknown): InspectorEntitySnapshot {
  const keys = path.split('.');
  const clone: InspectorEntitySnapshot = structuredClone(snapshot);
  let target: Record<string, unknown> = clone as unknown as Record<string, unknown>;
  for (const key of keys.slice(0, -1)) {
    target = target[key] as Record<string, unknown>;
  }
  target[keys[keys.length - 1]] = value;
  return clone;
}

export const schemaForSnapshot = (snapshot: InspectorEntitySnapshot) => inspectorComponentSchemas.filter((schema) => (
  schema.id === 'transform' ? snapshot.transform !== null : snapshot.camera !== null
));

