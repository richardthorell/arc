import { MoreVertical } from 'lucide-react';
import type { ReactNode } from 'react';
import { useEffect, useRef, useState } from 'react';

import { UiButton, UiContextMenu, UiContextMenuItem, UiIconButton, UiPropertyCard, UiSelect, UiTextInput } from '../ui';
import type { UiPropertyCardField } from '../ui';
import type { AssetPickerItem, AssetThumbnailProvider } from './AssetPicker';
import { AssetPicker, AssetPreview, MaterialPicker, PrefabPicker, TexturePicker } from './AssetPicker';
import { MaterialParameterSubsection } from './MaterialParameterSubsection';
import { ColorControl, NumberControl, NumberControlLabel, Vector3Control } from './InspectorControls';
import type { InspectorProceduralMesh, Vec3, Vec4 } from './inspectorTypes';
import { getPathValue } from './propertySchema';
import type { PropertyComponentSchema, PropertyFieldSchema, VectorAxis } from './propertySchema';

const meshAssignmentPrefix = '__arc_mesh__/';
const primitiveAssignmentPrefix = '__arc_primitive__/';
const primitiveParameterPrefix = '__arc_primitive_parameter__/';
const primitiveMeshUriPrefix = 'arc://primitive/';
const meshAssetExtensions = ['.glb', '.gltf', '.fbx', '.obj'] as const;
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

  const propertyFields: UiPropertyCardField[] = [];

  if (showMeshAsset) {
    propertyFields.push({
      id: '__mesh',
      label: 'Mesh',
      align: 'start',
      control: (
        <AssetPicker
          allowEmpty={false}
          allowedExtensions={meshAssetExtensions}
          assetKinds={['mesh', 'scene']}
          assetTypeLabel="Mesh"
          assets={meshAssets}
          label="Mesh"
          mixed={mixedFields.includes('meshRenderer.meshPath')}
          showLabel={false}
          thumbnailProvider={thumbnailProvider}
          value={(getPathValue(context, 'meshRenderer.meshPath') as string) || ''}
          onChange={(path) => {
            const assignment = path.startsWith(primitiveMeshUriPrefix)
              ? `${primitiveAssignmentPrefix}${path.slice(primitiveMeshUriPrefix.length)}`
              : `${meshAssignmentPrefix}${path}`;
            onValue('meshRenderer.materialPath', assignment, true);
          }}
        />
      ),
    });
  }

  if (proceduralMesh && selectionCount === 1) {
    propertyFields.push(
      ...proceduralMeshPropertyFields(proceduralMesh, (parameter, value, settled) =>
        onValue('meshRenderer.materialPath', `${primitiveParameterPrefix}${parameter}/${value}`, settled),
      ),
    );
  }

  for (const field of visibleFields) {
    const linked = field.type === 'vector3' && Boolean(field.linked) && !unlinkedFields.has(field.path);
    const value = getPathValue(context, field.path);
    const fieldValue = (next: unknown, settled: boolean) => onValue(field.path, next, settled);
    const label =
      field.type === 'number' ? (
        <NumberControlLabel
          field={field}
          value={value as number}
          onCommit={(next) => fieldValue(next, true)}
          onPreview={(next) => fieldValue(next, false)}
        />
      ) : (
        field.label
      );

    propertyFields.push({
      id: field.id,
      label,
      align: field.type === 'asset' || field.type === 'assetPreview' ? 'start' : 'center',
      tooltip: field.tooltip,
      control: (
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
          onValue={fieldValue}
          onAction={(action) => onAction?.(action)}
        />
      ),
    });

    if (field.type === 'asset' && field.assetKind === 'material') {
      const mixed = mixedFields.some((path) => path === field.path || path.startsWith(`${field.path}.`));
      const materialValue = (value as string) || '';
      const selectedMaterial = assets.find(
        (asset) =>
          asset.kind === 'material' &&
          (field.referenceMode === 'guid' ? (asset.guid || asset.id) === materialValue : asset.path === materialValue),
      );
      const canShowMaterialParameters =
        Boolean(materialValue) &&
        !mixed &&
        selectedMaterial?.scope !== 'procedural' &&
        Boolean(selectedMaterial || (field.referenceMode !== 'guid' && /\.arcmat$/i.test(materialValue)));

      if (canShowMaterialParameters) {
        propertyFields.push({
          id: `${field.id}-parameters`,
          fullWidth: true,
          className: 'inspector-material-parameter-row',
          control: (
            <MaterialParameterSubsection
              assets={assets}
              referenceMode={field.referenceMode}
              value={materialValue}
            />
          ),
        });
      }
    }
  }

  return (
    <UiPropertyCard
      actions={
        <>
          {headerAccessory && <div className="inspector-component-header-accessory">{headerAccessory}</div>}
          {onAction && (
            <UiIconButton
              aria-expanded={actionsOpen}
              aria-haspopup="menu"
              label={schema.title + ' component actions'}
              onClick={() => setActionsOpen((value) => !value)}
              type="button"
            >
              <MoreVertical size={15} />
            </UiIconButton>
          )}
          {onAction && actionsOpen && (
            <UiContextMenu
              aria-label={schema.title + ' component actions menu'}
              style={{ left: 'auto', right: '4px', top: 'calc(100% + 2px)', maxWidth: 'calc(100% - 8px)' }}
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
        </>
      }
      className="inspector-schema-component"
      collapsed={collapsed}
      contentClassName="inspector-component-content"
      emptyState={<div className="inspector-component-empty">No settings are active for this mode.</div>}
      fields={propertyFields}
      onToggle={onToggle}
      ref={componentRef}
      title={schema.title}
    />
  );
}

function proceduralMeshPropertyFields(
  mesh: InspectorProceduralMesh,
  onValue: (parameter: string, value: number, settled: boolean) => void,
): UiPropertyCardField[] {
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
  const fields: UiPropertyCardField[] = [
    {
      id: '__procedural-heading',
      label: `Procedural Mesh · ${mesh.type}`,
      fullWidth: true,
      className: 'inspector-procedural-mesh-heading',
    },
  ];

  const add = (
    parameter: keyof Omit<InspectorProceduralMesh, 'type'>,
    field: ReturnType<typeof dimensionField>,
  ) => {
    const value = mesh[parameter];
    if (typeof value !== 'number') return;
    fields.push({
      id: `__procedural-${parameter}`,
      label: (
        <NumberControlLabel
          field={field}
          value={value}
          onCommit={(next) => onValue(parameter, next, true)}
          onPreview={(next) => onValue(parameter, next, false)}
        />
      ),
      control: (
        <NumberControl
          field={field}
          showLabel={false}
          value={value}
          onCommit={(next) => onValue(parameter, next, true)}
          onPreview={(next) => onValue(parameter, next, false)}
        />
      ),
    });
  };

  if (mesh.type === 'plane') {
    add('width', dimensionField('Width'));
    add('depth', dimensionField('Depth'));
    add('segmentsX', segmentField('Segments X', 1));
    add('segmentsZ', segmentField('Segments Z', 1));
  } else if (mesh.type === 'cube') {
    add('width', dimensionField('Width'));
    add('height', dimensionField('Height'));
    add('depth', dimensionField('Depth'));
    add('segmentsX', segmentField('Segments X', 1));
    add('segmentsY', segmentField('Segments Y', 1));
    add('segmentsZ', segmentField('Segments Z', 1));
  } else if (mesh.type === 'sphere') {
    add('radius', dimensionField('Radius'));
    add('segments', segmentField('Segments', 3));
    add('rings', segmentField('Rings', 2));
  } else if (mesh.type === 'cylinder' || mesh.type === 'cone') {
    add('radius', dimensionField('Radius'));
    add('height', dimensionField('Height'));
    add('radialSegments', segmentField('Radial Segments', 3));
    add('heightSegments', segmentField('Height Segments', 1));
  } else if (mesh.type === 'capsule') {
    add('radius', dimensionField('Radius'));
    add('height', dimensionField('Height'));
    add('radialSegments', segmentField('Radial Segments', 3));
    add('hemisphereRings', segmentField('Hemisphere Rings', 2));
    add('heightSegments', segmentField('Height Segments', 1));
  }

  return fields;
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
        showLabel={false}
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
        showLabel={false}
        value={value as number}
        onCommit={(next) => onValue(next, true)}
        onPreview={(next) => onValue(next, false)}
      />
    );
  }
  if (field.type === 'boolean') {
    return (
      <input
        aria-label={field.ariaLabel ?? field.label}
        checked={mixed ? false : (value as boolean)}
        ref={(input) => {
          if (input) input.indeterminate = mixed;
        }}
        onChange={(event) => onValue(event.target.checked, true)}
        type="checkbox"
      />
    );
  }

  if (field.type === 'text') {
    return (
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
    );
  }
  if (field.type === 'enum') {
    const options = [
      ...(mixed ? [{ label: 'Mixed', value: '' }] : []),
      ...field.options.map((option) => ({ label: option.label, value: option.value })),
    ];
    return (
        <UiSelect
          ariaLabel={field.ariaLabel ?? field.label}
          options={options}
          value={mixed ? '' : (value as string)}
          onValueChange={(next) => onValue(next, true)}
        />
    );
  }
  if (field.type === 'asset') {
    if (field.assetKind === 'asset') {
      return (
        <AssetPicker
          allowEmpty={field.allowEmpty}
          allowedExtensions={field.allowedExtensions}
          assetKinds={['scene', 'mesh', 'material', 'texture', 'shader', 'prefab', 'water']}
          assetTypeIds={field.assetTypeId ? [field.assetTypeId] : undefined}
          assetTypeLabel="Asset"
          assets={assets}
          label={field.label}
          mixed={mixed}
          referenceMode={field.referenceMode}
          showLabel={false}
          thumbnailProvider={thumbnailProvider}
          value={(value as string) || ''}
          onChange={(next) => onValue(next, true)}
        />
      );
    }
    if (field.assetKind === 'material') {
      return (
        <MaterialPicker
          allowedExtensions={field.allowedExtensions}
          allowEmpty={field.allowEmpty}
          assets={assets}
          label={field.label}
          mixed={mixed}
          referenceMode={field.referenceMode}
          showLabel={false}
          showParameters={false}
          thumbnailProvider={thumbnailProvider}
          value={(value as string) || ''}
          onChange={(next) => onValue(next, true)}
        />
      );
    }
    const Picker = field.assetKind === 'prefab' ? PrefabPicker : TexturePicker;
    return (
      <Picker
        allowedExtensions={field.allowedExtensions}
        allowEmpty={field.allowEmpty}
        assets={assets}
        label={field.label}
        mixed={mixed}
        referenceMode={field.referenceMode}
        showLabel={false}
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
    return <output className="inspector-readonly-value" aria-label={field.ariaLabel ?? field.label}>{display}</output>;
  }
  if (field.type === 'actions') {
    return (
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
      showLabel={false}
      value={rgba}
      onCommit={(next) => onValue(colorValue(next), true)}
      onPreview={(next) => onValue(colorValue(next), false)}
    />
  );
}
