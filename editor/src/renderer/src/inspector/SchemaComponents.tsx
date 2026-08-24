import { ChevronDown, ChevronRight, MoreVertical } from 'lucide-react';
import type { ReactNode } from 'react';
import { useEffect, useRef, useState } from 'react';

import { UiButton, UiContextMenu, UiContextMenuItem, UiIconButton, UiSelect, UiTextInput } from '../ui';
import type { AssetPickerItem, AssetThumbnailProvider } from './AssetPicker';
import { AssetPicker, AssetPreview, MaterialPicker, PrefabPicker, TexturePicker } from './AssetPicker';
import { ColorControl, NumberControl, Vector3Control } from './InspectorControls';
import type { InspectorProceduralMesh, Vec3, Vec4 } from './inspectorTypes';
import { getPathValue } from './propertySchema';
import type { PropertyComponentSchema, PropertyFieldSchema, VectorAxis } from './propertySchema';

const meshAssignmentPrefix = '__arc_mesh__/';
const primitiveAssignmentPrefix = '__arc_primitive__/';
const primitiveParameterPrefix = '__arc_primitive_parameter__/';
const primitiveMeshUriPrefix = 'arc://primitive/';
const meshAssetExtensions = ['.glb', '.gltf', '.fbx'] as const;
const proceduralMeshAssets: ReadonlyArray<AssetPickerItem> = [
  {
    id: 'arc-primitive-plane',
    name: 'Plane',
    path: `${primitiveMeshUriPrefix}plane`,
    kind: 'mesh',
    status: 'ready',
    scope: 'procedural',
    readOnly: true,
  },
  {
    id: 'arc-primitive-cube',
    name: 'Cube',
    path: `${primitiveMeshUriPrefix}cube`,
    kind: 'mesh',
    status: 'ready',
    scope: 'procedural',
    readOnly: true,
  },
  {
    id: 'arc-primitive-sphere',
    name: 'Sphere',
    path: `${primitiveMeshUriPrefix}sphere`,
    kind: 'mesh',
    status: 'ready',
    scope: 'procedural',
    readOnly: true,
  },
  {
    id: 'arc-primitive-cylinder',
    name: 'Cylinder',
    path: `${primitiveMeshUriPrefix}cylinder`,
    kind: 'mesh',
    status: 'ready',
    scope: 'procedural',
    readOnly: true,
  },
  {
    id: 'arc-primitive-cone',
    name: 'Cone',
    path: `${primitiveMeshUriPrefix}cone`,
    kind: 'mesh',
    status: 'ready',
    scope: 'procedural',
    readOnly: true,
  },
  {
    id: 'arc-primitive-capsule',
    name: 'Capsule',
    path: `${primitiveMeshUriPrefix}capsule`,
    kind: 'mesh',
    status: 'ready',
    scope: 'procedural',
    readOnly: true,
  },
];

export function SchemaComponentCard<TContext extends object>({
  schema,
  context,
  collapsed,
  assets = [],
  thumbnailProvider,
  onToggle,
  onValue,
  onAction,
  headerAccessory,
}: {
  schema: PropertyComponentSchema<TContext>;
  context: TContext;
  collapsed: boolean;
  assets?: ReadonlyArray<AssetPickerItem>;
  thumbnailProvider?: AssetThumbnailProvider;
  onToggle: () => void;
  onValue: (path: string, value: unknown, settled: boolean) => void;
  onAction?: (action: string) => void;
  headerAccessory?: ReactNode;
}) {
  const [unlinkedFields, setUnlinkedFields] = useState<Set<string>>(() => new Set());
  const [actionsOpen, setActionsOpen] = useState(false);
  const componentRef = useRef<HTMLElement | null>(null);
  const visibleFields = schema.fields.filter((field) => !field.visible || field.visible(context));
  const mixedFields = (context as { aggregate?: { mixedFields?: string[] } }).aggregate?.mixedFields ?? [];
  const showMeshAsset = schema.id === 'meshRenderer' && !visibleFields.some((field) => field.id === 'mesh');
  const meshAssets = showMeshAsset
    ? [...proceduralMeshAssets, ...assets.filter((asset) => !asset.path.startsWith(primitiveMeshUriPrefix))]
    : assets;
  const proceduralMesh =
    schema.id === 'meshRenderer'
      ? ((context as { proceduralMesh?: InspectorProceduralMesh | null }).proceduralMesh ?? null)
      : null;
  const selectionCount = (context as { selectionCount?: number }).selectionCount ?? 1;

  useEffect(() => {
    if (!actionsOpen) return;

    const close = (event: PointerEvent) => {
      if (!componentRef.current?.contains(event.target as Node)) setActionsOpen(false);
    };

    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [actionsOpen]);

  const runComponentAction = (action: string) => {
    setActionsOpen(false);
    onAction?.(action);
  };

  return (
    <section ref={componentRef} className={`inspector-component-card ${collapsed ? 'is-collapsed' : ''}`}>
      <header
        style={{
          gridTemplateColumns: headerAccessory
            ? 'minmax(0, 1fr) auto var(--arc-icon-button-size)'
            : 'minmax(0, 1fr) var(--arc-icon-button-size)',
          minHeight: 'var(--arc-icon-button-size)',
        }}
      >
        <button aria-label={`${collapsed ? 'Expand' : 'Collapse'} ${schema.title}`} onClick={onToggle} type="button">
          {collapsed ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
          <span>{schema.title}</span>
        </button>
        {headerAccessory && <div className="inspector-component-header-accessory">{headerAccessory}</div>}
        {onAction && (
          <UiIconButton
            aria-expanded={actionsOpen}
            aria-haspopup="menu"
            label={`${schema.title} component actions`}
            onClick={() => setActionsOpen((value) => !value)}
            type="button"
          >
            <MoreVertical size={15} />
          </UiIconButton>
        )}
        {onAction && actionsOpen && (
          <UiContextMenu
            aria-label={`${schema.title} component actions menu`}
            style={{
              left: 'auto',
              right: '4px',
              top: 'calc(100% + 2px)',
              maxWidth: 'calc(100% - 8px)',
            }}
          >
            <UiContextMenuItem onClick={() => runComponentAction('copy')} type="button">
              Copy Component
            </UiContextMenuItem>
            <UiContextMenuItem onClick={() => runComponentAction('paste')} type="button">
              Paste Component Values
            </UiContextMenuItem>
            <UiContextMenuItem onClick={() => runComponentAction('reset')} type="button">
              Reset Component
            </UiContextMenuItem>
            {schema.id !== 'transform' && schema.id !== 'prefab' && (
              <UiContextMenuItem
                onClick={() => runComponentAction('remove')}
                style={{ color: 'var(--arc-color-danger)' }}
                type="button"
              >
                Remove Component
              </UiContextMenuItem>
            )}
          </UiContextMenu>
        )}
      </header>
      {!collapsed && (
        <div className="inspector-component-content">
          {showMeshAsset && (
            <AssetPicker
              allowEmpty={false}
              allowedExtensions={meshAssetExtensions}
              assetKinds={['mesh', 'scene']}
              assetTypeLabel="Mesh"
              assets={meshAssets}
              label="Mesh"
              mixed={mixedFields.includes('meshRenderer.meshPath')}
              thumbnailProvider={thumbnailProvider}
              value={(getPathValue(context, 'meshRenderer.meshPath') as string) || ''}
              onChange={(path) => {
                const assignment = path.startsWith(primitiveMeshUriPrefix)
                  ? `${primitiveAssignmentPrefix}${path.slice(primitiveMeshUriPrefix.length)}`
                  : `${meshAssignmentPrefix}${path}`;
                onValue('meshRenderer.materialPath', assignment, true);
              }}
            />
          )}
          {proceduralMesh && selectionCount === 1 && (
            <ProceduralMeshControls
              mesh={proceduralMesh}
              onValue={(parameter, value, settled) =>
                onValue('meshRenderer.materialPath', `${primitiveParameterPrefix}${parameter}/${value}`, settled)
              }
            />
          )}
          {visibleFields.map((field) => {
            const linked = field.type === 'vector3' && Boolean(field.linked) && !unlinkedFields.has(field.path);
            return (
              <div className="inspector-schema-field" key={field.id} title={field.tooltip}>
                <SchemaField
                  assets={assets}
                  context={context}
                  field={field}
                  linked={linked}
                  thumbnailProvider={thumbnailProvider}
                  onToggleLinked={() =>
                    setUnlinkedFields((current) => {
                      const next = new Set(current);
                      if (next.has(field.path)) next.delete(field.path);
                      else next.add(field.path);
                      return next;
                    })
                  }
                  onValue={(value, settled) => onValue(field.path, value, settled)}
                  onAction={(action) => onAction?.(action)}
                />
              </div>
            );
          })}
          {!visibleFields.length && !showMeshAsset && (
            <div className="inspector-component-empty">No settings are active for this mode.</div>
          )}
        </div>
      )}
    </section>
  );
}

function ProceduralMeshControls({
  mesh,
  onValue,
}: {
  mesh: InspectorProceduralMesh;
  onValue: (parameter: string, value: number, settled: boolean) => void;
}) {
  const dimensionField = (label: string) => ({
    label,
    precision: 3,
    step: 0.1,
    scrubSensitivity: 0.01,
    min: 0.001,
    max: 100000,
  });
  const segmentField = (label: string, min: number) => ({
    label,
    precision: 0,
    step: 1,
    scrubSensitivity: 0.25,
    min,
    max: 512,
  });
  const control = (
    parameter: keyof Omit<InspectorProceduralMesh, 'type'>,
    field: ReturnType<typeof dimensionField>,
  ) => {
    const value = mesh[parameter];
    if (typeof value !== 'number') return null;
    return (
      <NumberControl
        key={parameter}
        field={field}
        value={value}
        onCommit={(next) => onValue(parameter, next, true)}
        onPreview={(next) => onValue(parameter, next, false)}
      />
    );
  };

  return (
    <div className="inspector-procedural-mesh-controls">
      <div
        className="inspector-subsection-title"
        style={{
          marginTop: 4,
          padding: '7px 0 2px',
          borderTop: '1px solid var(--inspector-border-soft)',
          color: 'var(--inspector-text-muted)',
          fontSize: 10,
          fontWeight: 600,
          letterSpacing: '0.04em',
          textTransform: 'uppercase',
        }}
      >
        Procedural Mesh · {mesh.type}
      </div>
      {mesh.type === 'plane' && (
        <>
          {control('width', dimensionField('Width'))}
          {control('depth', dimensionField('Depth'))}
          {control('segmentsX', segmentField('Segments X', 1))}
          {control('segmentsZ', segmentField('Segments Z', 1))}
        </>
      )}
      {mesh.type === 'cube' && (
        <>
          {control('width', dimensionField('Width'))}
          {control('height', dimensionField('Height'))}
          {control('depth', dimensionField('Depth'))}
          {control('segmentsX', segmentField('Segments X', 1))}
          {control('segmentsY', segmentField('Segments Y', 1))}
          {control('segmentsZ', segmentField('Segments Z', 1))}
        </>
      )}
      {mesh.type === 'sphere' && (
        <>
          {control('radius', dimensionField('Radius'))}
          {control('segments', segmentField('Segments', 3))}
          {control('rings', segmentField('Rings', 2))}
        </>
      )}
      {(mesh.type === 'cylinder' || mesh.type === 'cone') && (
        <>
          {control('radius', dimensionField('Radius'))}
          {control('height', dimensionField('Height'))}
          {control('radialSegments', segmentField('Radial Segments', 3))}
          {control('heightSegments', segmentField('Height Segments', 1))}
        </>
      )}
      {mesh.type === 'capsule' && (
        <>
          {control('radius', dimensionField('Radius'))}
          {control('height', dimensionField('Height'))}
          {control('radialSegments', segmentField('Radial Segments', 3))}
          {control('hemisphereRings', segmentField('Hemisphere Rings', 2))}
          {control('heightSegments', segmentField('Height Segments', 1))}
        </>
      )}
    </div>
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
      if (Math.abs(source) < 1e-6) return { ...vector, [axis]: nextValue };
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
        onReset={field.resetValue === undefined ? undefined : () => onValue(structuredClone(field.resetValue), true)}
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
