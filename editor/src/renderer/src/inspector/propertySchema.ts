import type { Vec3, Vec4 } from './inspectorTypes';

export type VectorAxis = keyof Vec3;
export type ColorChannel = keyof Vec4;

type FieldBase<TContext> = {
  id: string;
  label: string;
  ariaLabel?: string;
  path: string;
  visible?: (context: TContext) => boolean;
  tooltip?: string;
};

export type Vector3FieldSchema<TContext = object> = FieldBase<TContext> & {
  type: 'vector3';
  precision: number;
  step: number;
  scrubSensitivity: number;
  unit?: string;
  linked?: boolean;
};

export type NumberFieldSchema<TContext = object> = FieldBase<TContext> & {
  type: 'number';
  precision: number;
  step: number;
  scrubSensitivity: number;
  unit?: string;
  min?: number;
  max?: number;
};

export type BooleanFieldSchema<TContext = object> = FieldBase<TContext> & { type: 'boolean' };
export type EnumFieldSchema<TContext = object> = FieldBase<TContext> & {
  type: 'enum';
  options: ReadonlyArray<{ value: string; label: string }>;
};
export type ColorFieldSchema<TContext = object> = FieldBase<TContext> & {
  type: 'color';
  precision: number;
  min: number;
  max: number;
  alpha?: boolean;
};
export type AssetReferenceFieldSchema<TContext = object> = FieldBase<TContext> & {
  type: 'asset';
  assetKind: 'texture' | 'material' | 'prefab';
  allowedExtensions?: ReadonlyArray<string>;
  allowEmpty?: boolean;
};
export type AssetPreviewFieldSchema<TContext = object> = FieldBase<TContext> & {
  type: 'assetPreview';
  assetKind: 'material';
  namePath?: string;
};
export type ReadonlyFieldSchema<TContext = object> = FieldBase<TContext> & {
  type: 'readonly';
  format?: (value: unknown, context: TContext) => string;
};
export type ActionFieldSchema<TContext = object> = FieldBase<TContext> & {
  type: 'actions';
  actions: ReadonlyArray<{
    id: string;
    label: string;
    danger?: boolean;
    disabled?: (context: TContext) => boolean;
    tooltip?: string;
  }>;
};

export type PropertyFieldSchema<TContext = object> =
  | Vector3FieldSchema<TContext>
  | NumberFieldSchema<TContext>
  | BooleanFieldSchema<TContext>
  | EnumFieldSchema<TContext>
  | ColorFieldSchema<TContext>
  | AssetReferenceFieldSchema<TContext>
  | AssetPreviewFieldSchema<TContext>
  | ReadonlyFieldSchema<TContext>
  | ActionFieldSchema<TContext>;

export type PropertyComponentSchema<TContext = object, TId extends string = string> = {
  id: TId;
  title: string;
  fields: ReadonlyArray<PropertyFieldSchema<TContext>>;
  visible?: (context: TContext) => boolean;
  collapsedByDefault?: boolean;
  badge?: string;
};

export function getPathValue<TContext extends object>(context: TContext, path: string): unknown {
  return path.split('.').reduce<unknown>((value, key) => (
    value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : undefined
  ), context);
}

export function setPathValue<TContext extends object>(context: TContext, path: string, value: unknown): TContext {
  const keys = path.split('.');
  const clone = structuredClone(context);
  let target = clone as unknown as Record<string, unknown>;
  for (const key of keys.slice(0, -1)) target = target[key] as Record<string, unknown>;
  target[keys[keys.length - 1]] = value;
  return clone;
}
