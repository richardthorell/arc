import { useEffect, useMemo, useRef, useState } from 'react';
import { Filter, Globe2, MoreVertical, Search } from 'lucide-react';

import type { SceneEntity } from '../services/editorHostTypes';
import { hostAssetReference } from '../services/assetReferences';
import { UiPanel } from '../ui';
import type { AssetPickerItem, AssetThumbnailProvider } from './AssetPicker';
import { AddComponentPicker } from './AddComponentPicker';

import { schemaForSnapshot, setPathValue } from './componentSchemas';
import type { HostProjectComponentSchema, InspectorComponentId } from './componentSchemas';
import type { HostResponse, InspectorEntitySnapshot, Vec3 } from './inspectorTypes';
import { cameraHostPayload, hostEntityKey, lightHostPayload, transformHostPayload } from './inspectorTypes';
import { InspectorComponentCard } from './InspectorComponentCard';
import { SkeletonInspector } from './SkeletonInspector';
import { inspectorLocalToWorld, inspectorWorldToLocal } from './transformSpace';
import { waterPresetOverrideForPath } from './waterPresetOverrides';

import './inspector.css';

export type InspectorEditTransaction = { id: number; phase: 'begin' | 'update' | 'commit' | 'cancel'; label?: string };
export type InspectorCommand = (
  type: string,
  payload: Record<string, unknown>,
  edit?: InspectorEditTransaction,
) => Promise<HostResponse>;

export type InspectorPanelProps = {
  snapshot: InspectorEntitySnapshot | null;
  loading?: boolean;
  command: InspectorCommand;
  refresh: () => Promise<void>;
  onStatus?: (message: string) => void;
  assets?: ReadonlyArray<AssetPickerItem>;
  thumbnailProvider?: AssetThumbnailProvider;
  projectSchemas?: ReadonlyArray<HostProjectComponentSchema>;
  scene?: ReadonlyArray<SceneEntity>;
  coordinateSpace?: 'local' | 'world';
  onCoordinateSpaceChange?: (space: 'local' | 'world') => void;
  readOnly?: boolean;
  contextLabel?: string;
};

const knownTags = ['Untagged', 'Camera', 'Light', 'Mesh', 'Environment'];
const defaultLayerMask = 1;
const environmentLayerMask = 2;
const proceduralParameterPrefix = '__arc_primitive_parameter__/';

const defaultWaterShape = (bodyType: 'ocean' | 'lake' | 'river') => {
  if (bodyType === 'lake')
    return {
      shapeClosed: true,
      shapePoints: [
        { position: { x: -10, y: 0, z: -10 }, width: 0, depth: 2, flow: 0 },
        { position: { x: 10, y: 0, z: -10 }, width: 0, depth: 2, flow: 0 },
        { position: { x: 10, y: 0, z: 10 }, width: 0, depth: 2, flow: 0 },
        { position: { x: -10, y: 0, z: 10 }, width: 0, depth: 2, flow: 0 },
      ],
    };
  if (bodyType === 'river')
    return {
      shapeClosed: false,
      shapePoints: [
        { position: { x: -12, y: 0, z: 0 }, width: 6, depth: 1.5, flow: 1 },
        { position: { x: 0, y: 0, z: 0 }, width: 6, depth: 1.5, flow: 1 },
        { position: { x: 12, y: 0, z: 0 }, width: 6, depth: 1.5, flow: 1 },
      ],
    };
  return { shapeClosed: false, shapePoints: [] };
};

function entityPayload(snapshot: InspectorEntitySnapshot) {
  return (snapshot.selectionCount ?? 1) > 1
    ? { entity: snapshot.entity, applyToSelection: true }
    : { entity: snapshot.entity };
}

function TextCommitInput({
  ariaLabel,
  value,
  onCommit,
  list,
  disabled = false,
  placeholder,
}: {
  ariaLabel: string;
  value: string;
  onCommit: (value: string) => void;
  list?: string;
  disabled?: boolean;
  placeholder?: string;
}) {
  const [draft, setDraft] = useState(value);
  const cancelBlur = useRef(false);
  useEffect(() => setDraft(value), [value]);
  const commit = () => {
    if (cancelBlur.current) {
      cancelBlur.current = false;
      return;
    }
    const next = draft.trim();
    if (next && next !== value) onCommit(next);
    else setDraft(value);
  };
  return (
    <input
      aria-label={ariaLabel}
      className="inspector-text-commit"
      disabled={disabled}
      list={list}
      onBlur={commit}
      onChange={(event) => setDraft(event.target.value)}
      onFocus={(event) => event.currentTarget.select()}
      onKeyDown={(event) => {
        if (event.key === 'Enter') event.currentTarget.blur();
        if (event.key === 'Escape') {
          cancelBlur.current = true;
          setDraft(value);
          event.currentTarget.blur();
        }
      }}
      placeholder={placeholder}
      value={draft}
    />
  );
}

export function InspectorPanel({
  snapshot,
  loading,
  command,
  refresh,
  onStatus,
  assets = [],
  thumbnailProvider,
  projectSchemas = [],
  scene = [],
  coordinateSpace = 'local',
  onCoordinateSpaceChange,
  readOnly = false,
  contextLabel,
}: InspectorPanelProps) {
  const [draft, setDraft] = useState(snapshot);
  const [filter, setFilter] = useState('');
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const confirmed = useRef(snapshot);
  const revision = useRef(0);
  const nextTransactionId = useRef(1);
  const activeTransaction = useRef<{ id: number; key: string } | null>(null);
  const componentClipboard = useRef<{ component: InspectorComponentId; value: unknown } | null>(null);

  useEffect(() => {
    confirmed.current = snapshot;
    setDraft(snapshot);
    setError(null);
  }, [snapshot]);

  const runMutation = async (
    next: InspectorEntitySnapshot,
    type: string,
    payload: Record<string, unknown>,
    settled = true,
    transactionKey?: string,
    transactionLabel?: string,
  ) => {
    if (readOnly) {
      const message = `${contextLabel || 'This context'} is read-only`;
      setError(message);
      onStatus?.(message);
      return;
    }
    const requestRevision = ++revision.current;
    setDraft(next);
    setError(null);
    try {
      let edit: InspectorEditTransaction | undefined;
      if (transactionKey && !settled) {
        if (!activeTransaction.current) {
          activeTransaction.current = { id: nextTransactionId.current++, key: transactionKey };
          edit = { id: activeTransaction.current.id, phase: 'begin', label: transactionLabel };
        } else {
          edit = { id: activeTransaction.current.id, phase: 'update', label: transactionLabel };
        }
      } else if (transactionKey && settled && activeTransaction.current?.key === transactionKey) {
        edit = { id: activeTransaction.current.id, phase: 'commit', label: transactionLabel };
        activeTransaction.current = null;
      }
      const response = edit ? await command(type, payload, edit) : await command(type, payload);
      if (requestRevision !== revision.current) return;
      if (!response.succeeded) {
        setDraft(confirmed.current);
        const message = response.error || 'Inspector update failed';
        setError(message);
        onStatus?.(message);
        return;
      }
      if (settled) confirmed.current = next;
      onStatus?.('Inspector value updated');
      if (settled) await refresh();
    } catch (reason) {
      if (requestRevision !== revision.current) return;
      setDraft(confirmed.current);
      const message = reason instanceof Error ? reason.message : String(reason);
      setError(message);
      onStatus?.(message);
    }
  };

  const updateHeader = (next: InspectorEntitySnapshot, type: string, extra: Record<string, unknown>) => {
    void runMutation(next, type, { ...entityPayload(next), ...extra });
  };

  const updateComponent = (
    component: InspectorComponentId,
    path: string,
    next: InspectorEntitySnapshot,
    settled: boolean,
    selectedAsset?: AssetPickerItem,
  ) => {
    const transactionKey = `${component}:${path}`;
    const transactionLabel =
      component === 'transform'
        ? 'Transform Entity'
        : component === 'camera'
          ? 'Edit Camera'
          : component.endsWith('Light')
            ? 'Edit Light'
            : component === 'meshRenderer'
              ? 'Edit Mesh Renderer'
              : component === 'water'
                ? 'Edit Water'
                : component === 'flow'
                  ? 'Edit Flow'
                  : 'Edit Terrain';
    if (component === 'transform' && next.transform) {
      void runMutation(
        next,
        'entity.setTransform',
        {
          ...entityPayload(next),
          transform: transformHostPayload(next.transform),
        },
        settled,
        transactionKey,
        transactionLabel,
      );
    } else if (component === 'camera' && next.camera) {
      void runMutation(
        next,
        'entity.setCamera',
        {
          ...entityPayload(next),
          camera: cameraHostPayload(next.camera),
        },
        settled,
        transactionKey,
        transactionLabel,
      );
    } else if (component.endsWith('Light') && next.light) {
      void runMutation(
        next,
        'entity.setLight',
        {
          ...entityPayload(next),
          light: lightHostPayload(next.light),
        },
        settled,
        transactionKey,
        transactionLabel,
      );
    } else if (component === 'meshRenderer' && next.meshRenderer) {
      if (path === 'meshRenderer.materialPath') {
        const proceduralParameter = next.meshRenderer.materialPath.startsWith(proceduralParameterPrefix);
        void runMutation(
          next,
          'entity.setMaterial',
          {
            ...entityPayload(next),
            path: next.meshRenderer.materialPath,
            ...(selectedAsset ? { asset: hostAssetReference(selectedAsset) ?? undefined } : {}),
          },
          proceduralParameter ? settled : true,
          proceduralParameter ? transactionKey : undefined,
          proceduralParameter ? 'Edit Procedural Mesh' : undefined,
        );
      } else {
        void runMutation(
          next,
          'entity.setMeshRenderer',
          {
            ...entityPayload(next),
            representation: next.meshRenderer.representation,
            visible: next.meshRenderer.visible,
            castsShadows: next.meshRenderer.castsShadows,
            receivesShadows: next.meshRenderer.receivesShadows,
            shadowLodBias: next.meshRenderer.shadowLodBias,
            maximumShadowDistance: next.meshRenderer.maximumShadowDistance,
            motionVectors:
              next.meshRenderer.motionVectors === 'always' ? 1 : next.meshRenderer.motionVectors === 'disabled' ? 2 : 0,
            receiveDecals: next.meshRenderer.receiveDecals,
            occlusionCulling: next.meshRenderer.occlusionCulling,
            boundsScale: next.meshRenderer.boundsScale,
            minimumDrawDistance: next.meshRenderer.minimumDrawDistance,
            maximumDrawDistance: next.meshRenderer.maximumDrawDistance,
            forcedLod: next.meshRenderer.forcedLod,
            lodBias: next.meshRenderer.lodBias,
            affectsIndirectLighting: next.meshRenderer.affectsIndirectLighting,
            surfaceCardDensityBias: next.meshRenderer.surfaceCardDensityBias,
            distanceFieldResolutionBias: next.meshRenderer.distanceFieldResolutionBias,
            visibleInHardwareTracing: next.meshRenderer.visibleInHardwareTracing,
          },
          settled,
          transactionKey,
          transactionLabel,
        );
      }
    } else if (component === 'terrain' && next.terrain) {
      const layerMatch = /^terrain\.layers\.(\d)\.baseColorPath$/.exec(path);
      if (layerMatch) {
        void runMutation(
          next,
          'terrain.assignLayer',
          {
            ...entityPayload(next),
            layer: Number(layerMatch[1]),
            path: next.terrain.layers[Number(layerMatch[1])].baseColorPath,
            ...(selectedAsset ? { asset: hostAssetReference(selectedAsset) ?? undefined } : {}),
          },
          true,
        );
      } else {
        void runMutation(
          next,
          'terrain.update',
          {
            ...entityPayload(next),
            enabled: next.terrain.enabled,
            receiveShadows: next.terrain.receiveShadows,
            castShadows: next.terrain.castShadows,
            patchQuads: next.terrain.patchQuads,
            maximumHierarchyDepth: next.terrain.maximumHierarchyDepth,
            geometricErrorMultiplier: next.terrain.geometricErrorMultiplier,
            shadowLodBias: next.terrain.shadowLodBias,
            maximumShadowDistance: next.terrain.maximumShadowDistance,
          },
          settled,
          transactionKey,
          transactionLabel,
        );
      }
    } else if (component === 'flow' && next.flow) {
      const flowReference = selectedAsset ? hostAssetReference(selectedAsset) : null;
      const normalizedFlow =
        flowReference && path === 'flow.graphGuid'
          ? { ...next.flow, graphGuid: flowReference.guid, graphPathHint: flowReference.pathHint || '' }
          : next.flow;
      const normalizedNext =
        normalizedFlow === next.flow ? next : ({ ...next, flow: normalizedFlow } as InspectorEntitySnapshot);
      void runMutation(
        normalizedNext,
        'entity.setFlow',
        {
          ...entityPayload(normalizedNext),
          graphGuid: normalizedFlow.graphGuid,
          graphPathHint: normalizedFlow.graphPathHint,
          ...(flowReference ? { asset: flowReference } : {}),
          enabled: normalizedFlow.enabled,
        },
        settled,
        transactionKey,
        transactionLabel,
      );
    } else if (component === 'water' && next.water) {
      let normalizedWater = next.water;
      if (path === 'water.bodyType') {
        normalizedWater = {
          ...normalizedWater,
          ...defaultWaterShape(normalizedWater.bodyType),
          presetGuid: '',
          presetPath: '',
          presetOverrideMask: 0,
        };
      } else if (path === 'water.presetPath') {
        const presetReference = selectedAsset ? hostAssetReference(selectedAsset) : null;
        normalizedWater = {
          ...normalizedWater,
          presetGuid: presetReference?.guid || '',
          presetOverrideMask: next.water.presetPath
            ? draft?.water?.presetPath
              ? next.water.presetOverrideMask
              : 0
            : 0,
        };
      } else {
        const overrideBit = waterPresetOverrideForPath[path];
        if (overrideBit && normalizedWater.presetPath)
          normalizedWater = {
            ...normalizedWater,
            presetOverrideMask: normalizedWater.presetOverrideMask | overrideBit,
          };
      }
      const normalizedNext =
        normalizedWater === next.water ? next : ({ ...next, water: normalizedWater } as InspectorEntitySnapshot);
      const water = {
        ...normalizedWater,
        materialGuid:
          path === 'water.materialPath'
            ? selectedAsset
              ? hostAssetReference(selectedAsset)?.guid || ''
              : ''
            : normalizedWater.materialGuid,
        absorption: [normalizedWater.absorption.x, normalizedWater.absorption.y, normalizedWater.absorption.z],
        scattering: [normalizedWater.scattering.x, normalizedWater.scattering.y, normalizedWater.scattering.z],
        shapePoints: normalizedWater.shapePoints.map((point) => ({
          ...point,
          position: [point.position.x, point.position.y, point.position.z],
        })),
      };
      void runMutation(
        normalizedNext,
        'water.update',
        { ...entityPayload(normalizedNext), water },
        settled,
        transactionKey,
        transactionLabel,
      );
    } else {
      const match = /^projectComponents\.(\d+)\.values\.(.+)$/.exec(path);
      if (!match) return;
      const projectComponent = next.projectComponents[Number(match[1])];
      if (!projectComponent) return;
      void runMutation(
        next,
        'component.patchField',
        { component: projectComponent.typeId, field: match[2], value: projectComponent.values[match[2]] },
        settled,
        transactionKey,
        `Edit ${projectComponent.displayName}`,
      );
    }
  };

  const displayDraft = useMemo(() => {
    if (!draft?.transform || coordinateSpace !== 'world' || (draft.selectionCount ?? 1) > 1) return draft;
    return {
      ...draft,
      transform: inspectorLocalToWorld(scene, hostEntityKey(draft.entity), draft.transform),
    };
  }, [coordinateSpace, draft, scene]);

  const schemas = useMemo(() => {
    if (!displayDraft) return [];
    const needle = filter.trim().toLocaleLowerCase();
    const common = new Set(displayDraft.aggregate?.commonComponents ?? []);
    return schemaForSnapshot(displayDraft, projectSchemas).filter((schema) => {
      const componentKey = schema.id.endsWith('Light') ? 'light' : schema.id;
      if ((displayDraft.selectionCount ?? 1) > 1 && !common.has(componentKey)) return false;
      return (
        !needle ||
        schema.title.toLocaleLowerCase().includes(needle) ||
        schema.fields.some((field) => field.label.toLocaleLowerCase().includes(needle))
      );
    });
  }, [displayDraft, filter, projectSchemas]);

  const runComponentAction = (component: InspectorComponentId, action: string) => {
    if (!draft) return;
    if (component === 'water' && draft.water && action.startsWith('water.revertPresetOverride:')) {
      const path = action.slice('water.revertPresetOverride:'.length);
      const bit = waterPresetOverrideForPath[path];
      if (!bit) return;
      const next = {
        ...draft,
        water: { ...draft.water, presetOverrideMask: draft.water.presetOverrideMask & ~bit },
      };
      updateComponent('water', 'water.presetOverrideMask', next, true);
      return;
    }
    if (component === 'water' && draft.water && action === 'water.revertPresetOverrides') {
      const next = { ...draft, water: { ...draft.water, presetOverrideMask: 0 } };
      updateComponent('water', 'water.presetOverrideMask', next, true);
      return;
    }
    const componentKey: Partial<Record<InspectorComponentId, keyof InspectorEntitySnapshot>> = {
      transform: 'transform',
      camera: 'camera',
      meshRenderer: 'meshRenderer',
      directionalLight: 'light',
      pointLight: 'light',
      spotLight: 'light',
      areaLight: 'light',
      terrain: 'terrain',
      water: 'water',
      flow: 'flow',
      prefab: 'prefab',
    };
    const key = componentKey[component];
    if (action === 'copy' && key) {
      componentClipboard.current = { component, value: structuredClone(draft[key]) };
      onStatus?.(`${component} copied`);
      return;
    }
    if (action === 'paste') {
      const copied = componentClipboard.current;
      if (!key || !copied || copied.component !== component) {
        setError('The component clipboard does not contain a compatible component.');
        return;
      }
      const next = { ...draft, [key]: structuredClone(copied.value) } as InspectorEntitySnapshot;
      updateComponent(component, String(key), next, true);
      return;
    }
    if (action === 'reset' || action === 'remove') {
      void (async () => {
        const response = await command(`component.${action}`, { component });
        if (!response.succeeded) {
          setError(response.error || `Could not ${action} ${component}`);
          return;
        }
        onStatus?.(`${component} ${action === 'reset' ? 'reset' : 'removed'}`);
        await refresh();
      })();
      return;
    }
    if (component !== 'prefab') return;
    const commandType =
      action === 'apply'
        ? 'prefab.apply'
        : action === 'revert'
          ? 'prefab.revert'
          : action === 'unpack'
            ? 'prefab.unpack'
            : '';
    if (!commandType) return;
    void (async () => {
      setError(null);
      try {
        const response = await command(commandType, entityPayload(draft));
        if (!response.succeeded) {
          const message = response.error || `Prefab ${action} failed`;
          setError(message);
          onStatus?.(message);
          return;
        }
        onStatus?.(`Prefab ${action} completed`);
        await refresh();
      } catch (reason) {
        const message = reason instanceof Error ? reason.message : String(reason);
        setError(message);
        onStatus?.(message);
      }
    })();
  };

  if (loading && !draft) return <div className="inspector-state">Loading selection…</div>;
  if (!draft) return <div className="inspector-state">Select an entity to inspect its components.</div>;

  const layerValue = draft.aggregate?.mixedFields.includes('renderLayerMask')
    ? 'mixed'
    : draft.renderLayerMask === defaultLayerMask
      ? String(defaultLayerMask)
      : draft.renderLayerMask === environmentLayerMask
        ? String(environmentLayerMask)
        : `custom:${draft.renderLayerMask}`;
  const tagOptions = knownTags.includes(draft.tag || 'Untagged') ? knownTags : [...knownTags, draft.tag];
  const activeMixed = draft.aggregate?.mixedFields.includes('active') ?? false;
  const tagMixed = draft.aggregate?.mixedFields.includes('tag') ?? false;
  const mobilityMixed = draft.aggregate?.mixedFields.includes('mobility') ?? false;

  return (
    <UiPanel className="data-inspector" variant="inspector">
      {readOnly && (
        <div className="inspector-read-only-notice" role="status">
          <strong>{contextLabel || 'Read-only inspection'}</strong>
          <span>Runtime values can be inspected, but authoring edits are disabled and discarded when Play stops.</span>
        </div>
      )}
      <fieldset className="inspector-read-only-scope" disabled={readOnly}>
        <header className="inspector-entity-card">
          <div className="inspector-entity-title-row">
            <input
              aria-label="Entity active"
              checked={activeMixed ? false : draft.active}
              ref={(input) => {
                if (input) input.indeterminate = activeMixed;
              }}
              onChange={(event) =>
                updateHeader({ ...draft, active: event.target.checked }, 'entity.setActive', {
                  active: event.target.checked,
                })
              }
              type="checkbox"
            />
            <TextCommitInput
              ariaLabel="Entity name"
              disabled={(draft.selectionCount ?? 1) > 1}
              value={(draft.selectionCount ?? 1) > 1 ? `${draft.selectionCount} entities selected` : draft.name}
              onCommit={(name) => updateHeader({ ...draft, name }, 'entity.rename', { name })}
            />
            <label className="inspector-static" title={`Mobility: ${draft.mobility ?? 'movable'}`}>
              <input
                aria-label="Static"
                checked={draft.mobility === 'static'}
                ref={(input) => {
                  if (input) input.indeterminate = mobilityMixed || draft.mobility === 'stationary';
                }}
                onChange={(event) => {
                  const mobility = event.target.checked ? 'static' : 'movable';
                  updateHeader({ ...draft, mobility }, 'entity.setMobility', { mobility });
                }}
                type="checkbox"
              />
              <span>Static</span>
            </label>
            <button aria-label="Entity actions" className="inspector-menu-button" type="button">
              <MoreVertical size={15} />
            </button>
          </div>
          <div className="inspector-entity-meta-row">
            <label>
              <span>Tag</span>
              <TextCommitInput
                ariaLabel="Tag"
                list="arc-inspector-tags"
                placeholder={tagMixed ? 'Mixed' : undefined}
                value={tagMixed ? '' : draft.tag || 'Untagged'}
                onCommit={(value) => {
                  const tag = value === 'Untagged' ? '' : value;
                  updateHeader({ ...draft, tag }, 'entity.setTag', { tag });
                }}
              />
              <datalist id="arc-inspector-tags">
                {tagOptions.map((tag) => (
                  <option key={tag} value={tag} />
                ))}
              </datalist>
            </label>
            <label>
              <span>Layer</span>
              <select
                aria-label="Layer"
                value={layerValue}
                onChange={(event) => {
                  if (event.target.value === 'mixed' || event.target.value.startsWith('custom:')) return;
                  const renderLayerMask = Number(event.target.value);
                  updateHeader({ ...draft, renderLayerMask }, 'entity.setRenderLayer', { renderLayerMask });
                }}
              >
                {layerValue === 'mixed' && <option value="mixed">Mixed</option>}
                <option value={String(defaultLayerMask)}>Default</option>
                <option value={String(environmentLayerMask)}>Environment</option>
                {layerValue.startsWith('custom:') && (
                  <option value={layerValue}>{`Custom (0x${draft.renderLayerMask.toString(16).toUpperCase()})`}</option>
                )}
              </select>
            </label>
          </div>
        </header>

        <div className="inspector-search-row">
          <label>
            <Search size={17} />
            <input
              aria-label="Search components"
              onChange={(event) => setFilter(event.target.value)}
              placeholder="Search components…"
              value={filter}
            />
          </label>
          <button aria-label="Component filter options" type="button">
            <Filter size={17} />
          </button>
        </div>

        {error && (
          <div className="inspector-error" role="alert">
            {error}
          </div>
        )}
        {(draft.aggregate?.partialComponents.length ?? 0) > 0 && (
          <div className="inspector-mixed-components">
            Partial components: {draft.aggregate?.partialComponents.join(', ')}. Add or remove them to edit together.
          </div>
        )}
        <div className="inspector-component-list">
          {draft.skeleton && <SkeletonInspector skeleton={draft.skeleton} command={command} onStatus={onStatus} />}
          {draft.prefab && (
            <section className="prefab-override-strip">
              <div>
                <strong>Prefab Instance</strong>
                <span>
                  {draft.prefab.overrideCount} override{draft.prefab.overrideCount === 1 ? '' : 's'}
                </span>
              </div>
              {draft.prefab.sourceMissing && <b>Source missing</b>}
              <details>
                <summary>Overrides</summary>
                <div className="prefab-override-list">
                  {draft.prefab.overrides.map((override) => (
                    <div key={`${override.sourceEntity}:${override.componentId}:${override.fieldId}:${override.kind}`}>
                      <span>
                        <b>{override.kind}</b>
                        <code>{override.componentId}</code>
                        <small>Field {override.fieldId}</small>
                      </span>
                      <button
                        onClick={() =>
                          void command('prefab.revertOverride', { entity: draft.entity, ...override }).then(refresh)
                        }
                      >
                        Revert
                      </button>
                    </div>
                  ))}
                  {!draft.prefab.overrides.length && <small>No authored overrides.</small>}
                </div>
              </details>
              <button
                aria-label="Apply all prefab overrides"
                onClick={() => void command('prefab.apply', { entity: draft.entity }).then(refresh)}
              >
                Apply All
              </button>
              <button
                aria-label="Revert all prefab overrides"
                onClick={() => void command('prefab.revert', { entity: draft.entity }).then(refresh)}
              >
                Revert All
              </button>
              <button
                aria-label="Unpack prefab from override strip"
                onClick={() => void command('prefab.unpack', { entity: draft.entity }).then(refresh)}
              >
                Unpack
              </button>
            </section>
          )}
          {schemas.map((schema) => (
            <InspectorComponentCard
              key={schema.id}
              collapsed={collapsed[schema.id] ?? false}
              context={displayDraft ?? draft}
              schema={schema}
              assets={assets}
              headerAccessory={
                schema.id === 'transform' ? (
                  <label
                    className="inspector-coordinate-space"
                    title="Choose whether Transform values are edited relative to the parent or in world space."
                  >
                    <Globe2 aria-hidden="true" size={12} />
                    <select
                      aria-label="Inspector transform coordinate space"
                      disabled={(draft.selectionCount ?? 1) > 1}
                      onChange={(event) => onCoordinateSpaceChange?.(event.target.value as 'local' | 'world')}
                      value={coordinateSpace}
                    >
                      <option value="local">Local</option>
                      <option value="world">World</option>
                    </select>
                  </label>
                ) : undefined
              }
              thumbnailProvider={thumbnailProvider}
              onToggle={() => setCollapsed((value) => ({ ...value, [schema.id]: !(value[schema.id] ?? false) }))}
              onAction={(action) => runComponentAction(schema.id, action)}
              onValue={(path, value, settled, selectedAsset) => {
                if (path === 'terrain.activeLayer') value = Number(value);
                let next: InspectorEntitySnapshot;
                if (
                  schema.id === 'transform' &&
                  coordinateSpace === 'world' &&
                  displayDraft?.transform &&
                  draft.transform &&
                  (draft.selectionCount ?? 1) === 1
                ) {
                  const nextDisplay = setPathValue(displayDraft, path, value);
                  next = {
                    ...draft,
                    transform: inspectorWorldToLocal(scene, hostEntityKey(draft.entity), nextDisplay.transform!),
                  };
                } else {
                  next = setPathValue(draft, path, value);
                }
                if (path === 'transform.rotationDegrees' && next.transform && coordinateSpace !== 'world') {
                  next = { ...next, transform: { ...next.transform, rotationDegrees: value as Vec3 } };
                }
                updateComponent(schema.id, path, next, settled, selectedAsset);
              }}
            />
          ))}
          {!schemas.length && <div className="inspector-state compact">No components match “{filter}”.</div>}
          <AddComponentPicker
            snapshot={draft}
            projectSchemas={projectSchemas}
            onAdd={async (component, label) => {
              const response = await command('component.add', { component });
              if (!response.succeeded) {
                setError(response.error || `Could not add ${label}`);
                return false;
              }
              onStatus?.(`${label} added`);
              await refresh();
              return true;
            }}
          />
        </div>
      </fieldset>
    </UiPanel>
  );
}
