import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';

import { UiButton, UiSelect, UiTextInput } from '../ui';
import type { AssetPickerItem, AssetThumbnailProvider } from './AssetPicker';
import { AssetPicker, AssetPreview, MaterialPicker, PrefabPicker, TexturePicker } from './AssetPicker';
import { ColorControl, NumberControl, Vector3Control } from './InspectorControls';
import type { Vec3, Vec4 } from './inspectorTypes';
import { getPathValue } from './propertySchema';
import type { PropertyComponentSchema, PropertyFieldSchema, VectorAxis } from './propertySchema';

export function SchemaComponentCard<TContext extends object>({
  schema,
  context,
  collapsed,
  assets = [],
  thumbnailProvider,
  onToggle,
  onValue,
  onAction,
}: {
  schema: PropertyComponentSchema<TContext>;
  context: TContext;
  collapsed: boolean;
  assets?: ReadonlyArray<AssetPickerItem>;
  thumbnailProvider?: AssetThumbnailProvider;
  onToggle: () => void;
  onValue: (path: string, value: unknown, settled: boolean) => void;
  onAction?: (action: string) => void;
}) {
  const [linked, setLinked] = useState(false);
  const [actionsOpen, setActionsOpen] = useState(false);
  const visibleFields = schema.fields.filter((field) => !field.visible || field.visible(context));
  return (
    <section className={`inspector-component-card ${collapsed ? 'is-collapsed' : ''}`}>
      <header>
        <button aria-label={`${collapsed ? 'Expand' : 'Collapse'} ${schema.title}`} onClick={onToggle} type="button">
          {collapsed ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
          <span>{schema.title}</span>
          {schema.badge && <small>{schema.badge}</small>}
        </button>
        <button
          aria-expanded={actionsOpen}
          aria-label={`${schema.title} component actions`}
          onClick={() => setActionsOpen((value) => !value)}
          type="button"
        >
          <ChevronDown size={15} />
        </button>
        {actionsOpen && (
          <div className="inspector-component-menu">
            <button onClick={() => onAction?.('copy')} type="button">
              Copy Component
            </button>
            <button onClick={() => onAction?.('paste')} type="button">
              Paste Component Values
            </button>
            <button onClick={() => onAction?.('reset')} type="button">
              Reset Component
            </button>
            {schema.id !== 'transform' && schema.id !== 'prefab' && (
              <button className="danger" onClick={() => onAction?.('remove')} type="button">
                Remove Component
              </button>
            )}
          </div>
        )}
      </header>
      {!collapsed && (
        <div className="inspector-component-content">
          {visibleFields.map((field) => (
            <SchemaField
              key={field.id}
              assets={assets}
              context={context}
              field={field}
              linked={linked}
              thumbnailProvider={thumbnailProvider}
              onToggleLinked={() => setLinked((value) => !value)}
              onValue={(value, settled) => onValue(field.path, value, settled)}
              onAction={(action) => onAction?.(action)}
            />
          ))}
          {!visibleFields.length && (
            <div className="inspector-component-empty">No settings are active for this mode.</div>
          )}
        </div>
      )}
    </section>
  );
}

function SchemaField<TContext extends object>({
  field,
  context,
  linked,
  assets,
  thumbnailProvider,
  onToggleLinked,
  onValue,
  onAction,
}: {
  field: PropertyFieldSchema<TContext>;
  context: TContext;
  linked: boolean;
  assets: ReadonlyArray<AssetPickerItem>;
  thumbnailProvider?: AssetThumbnailProvider;
  onToggleLinked: () => void;
  onValue: (value: unknown, settled: boolean) => void;
  onAction: (action: string) => void;
}) {
  const value = getPathValue(context, field.path);
  const aggregate = (context as { aggregate?: { mixedFields?: string[] } }).aggregate;
  const mixed =
    aggregate?.mixedFields?.some((path) => path === field.path || path.startsWith(`${field.path}.`)) ?? false;
  if (field.type === 'vector3') {
    const vector = value as Vec3;
    const updateAxis = (axis: VectorAxis, nextValue: number) => {
      if (!linked || !field.linked) return { ...vector, [axis]: nextValue };
      const source = vector[axis];
      if (Math.abs(source) < 1e-6) return { x: nextValue, y: nextValue, z: nextValue };
      const ratio = nextValue / source;
      return { x: vector.x * ratio, y: vector.y * ratio, z: vector.z * ratio };
    };
    return (
      <Vector3Control
        field={field}
        linked={linked}
        mixed={mixed}
        value={vector}
        onToggleLinked={onToggleLinked}
        onCommit={(axis, next) => onValue(updateAxis(axis, next), true)}
        onPreview={(axis, next) => onValue(updateAxis(axis, next), false)}
      />
    );
  }
  if (field.type === 'number') {
    return (
      <NumberControl
        field={field}
        mixed={mixed}
        value={value as number}
        onCommit={(next) => onValue(next, true)}
        onPreview={(next) => onValue(next, false)}
      />
    );
  }
  if (field.type === 'boolean') {
    return (
      <label className="inspector-property inspector-checkbox-property" title={field.tooltip}>
        <span className="inspector-property-label">{field.label}</span>
        <input
          aria-label={field.ariaLabel ?? field.label}
          checked={mixed ? false : (value as boolean)}
          ref={(input) => {
            if (input) input.indeterminate = mixed;
          }}
          onChange={(event) => onValue(event.target.checked, true)}
          type="checkbox"
        />
      </label>
    );
  }

  if (field.type === 'text') {
    return (
      <label className="inspector-property" title={field.tooltip}>
        <span className="inspector-property-label">{field.label}</span>
        <UiTextInput
          aria-label={field.ariaLabel ?? field.label}
          disabled={field.readOnly}
          value={mixed ? '' : typeof value === 'string' ? value : ''}
          onChange={(event) => onValue(event.target.value, false)}
          onBlur={(event) => onValue(event.target.value, true)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') event.currentTarget.blur();
          }}
        />
      </label>
    );
  }
  if (field.type === 'enum') {
    const options = [
      ...(mixed ? [{ label: 'Mixed', value: '' }] : []),
      ...field.options.map((option) => ({ label: option.label, value: option.value })),
    ];
    return (
      <label className="inspector-property" title={field.tooltip}>
        <span className="inspector-property-label">{field.label}</span>
        <UiSelect
          ariaLabel={field.ariaLabel ?? field.label}
          options={options}
          value={mixed ? '' : (value as string)}
          onValueChange={(next) => onValue(next, true)}
        />
      </label>
    );
  }
  if (field.type === 'asset') {
    if (field.assetKind === 'asset') {
      return (
        <AssetPicker
          allowEmpty={field.allowEmpty}
          assetKinds={['scene', 'mesh', 'material', 'texture', 'shader', 'prefab']}
          assetTypeIds={field.assetTypeId ? [field.assetTypeId] : undefined}
          assetTypeLabel="Asset"
          assets={assets}
          label={field.label}
          mixed={mixed}
          referenceMode={field.referenceMode}
          thumbnailProvider={thumbnailProvider}
          value={(value as string) || ''}
          onChange={(next) => onValue(next, true)}
        />
      );
    }
    const Picker =
      field.assetKind === 'material' ? MaterialPicker : field.assetKind === 'prefab' ? PrefabPicker : TexturePicker;
    return (
      <Picker
        allowedExtensions={field.allowedExtensions}
        allowEmpty={field.allowEmpty}
        assets={assets}
        label={field.label}
        mixed={mixed}
        referenceMode={field.referenceMode}
        thumbnailProvider={thumbnailProvider}
        value={(value as string) || ''}
        onChange={(next) => onValue(next, true)}
      />
    );
  }
  if (field.type === 'assetPreview') {
    const name = field.namePath ? (getPathValue(context, field.namePath) as string) : '';
    return <AssetPreview label={field.label} name={name} path={(value as string) || ''} provider={thumbnailProvider} />;
  }
  if (field.type === 'readonly') {
    const display = field.format ? field.format(value, context) : String(value ?? '');
    return (
      <div className="inspector-property inspector-readonly-property" title={field.tooltip}>
        <span className="inspector-property-label">{field.label}</span>
        <output aria-label={field.ariaLabel ?? field.label}>{display}</output>
      </div>
    );
  }
  if (field.type === 'actions') {
    return (
      <div className="inspector-property inspector-action-property">
        <span className="inspector-property-label">{field.label}</span>
        <div className="ui-button-group ui-button-group-fill">
          {field.actions.map((action) => (
            <UiButton
              aria-label={action.label}
              disabled={action.disabled?.(context)}
              key={action.id}
              onClick={() => onAction(action.id)}
              title={action.tooltip}
              type="button"
              variant={action.danger ? 'danger' : 'default'}
            >
              {action.label}
            </UiButton>
          ))}
        </div>
      </div>
    );
  }
  const source = value as Vec3 | Vec4;
  const hasAlpha = field.alpha !== false && 'w' in source;
  const rgba: Vec4 = { x: source.x, y: source.y, z: source.z, w: hasAlpha ? source.w : 1 };
  const colorValue = (next: Vec4) => (hasAlpha ? next : { x: next.x, y: next.y, z: next.z });
  return (
    <ColorControl
      label={field.label}
      mixed={mixed}
      showAlpha={hasAlpha}
      value={rgba}
      onCommit={(next) => onValue(colorValue(next), true)}
      onPreview={(next) => onValue(colorValue(next), false)}
    />
  );
}
