import { useEffect, useMemo, useRef, useState } from 'react';
import { Filter, MoreVertical, Search } from 'lucide-react';

import type { AssetPickerItem, AssetThumbnailProvider } from './AssetPicker';

import { schemaForSnapshot, setPathValue } from './componentSchemas';
import type { HostProjectComponentSchema, InspectorComponentId } from './componentSchemas';
import type { HostResponse, InspectorEntitySnapshot, Vec3 } from './inspectorTypes';
import { cameraHostPayload, lightHostPayload, transformHostPayload } from './inspectorTypes';
import { SchemaComponentCard } from './SchemaComponents';

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
};

const knownTags = ['Untagged', 'Camera', 'Light', 'Mesh', 'Environment'];
const defaultLayerMask = 1;
const environmentLayerMask = 2;

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
        void runMutation(
          next,
          'entity.setMaterial',
          {
            ...entityPayload(next),
            path: next.meshRenderer.materialPath,
          },
          true,
        );
      } else {
        const tint = next.meshRenderer.baseColorTint;
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
            baseColorTint: [tint.x, tint.y, tint.z, tint.w],
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

  const schemas = useMemo(() => {
    if (!draft) return [];
    const needle = filter.trim().toLocaleLowerCase();
    const common = new Set(draft.aggregate?.commonComponents ?? []);
    return schemaForSnapshot(draft, projectSchemas).filter((schema) => {
      const componentKey = schema.id.endsWith('Light') ? 'light' : schema.id;
      if ((draft.selectionCount ?? 1) > 1 && !common.has(componentKey)) return false;
      return (
        !needle ||
        schema.title.toLocaleLowerCase().includes(needle) ||
        schema.fields.some((field) => field.label.toLocaleLowerCase().includes(needle))
      );
    });
  }, [draft, filter, projectSchemas]);

  const runComponentAction = (component: InspectorComponentId, action: string) => {
    if (!draft) return;
    const componentKey: Partial<Record<InspectorComponentId, keyof InspectorEntitySnapshot>> = {
      transform: 'transform',
      camera: 'camera',
      meshRenderer: 'meshRenderer',
      directionalLight: 'light',
      pointLight: 'light',
      spotLight: 'light',
      areaLight: 'light',
      terrain: 'terrain',
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
    <section className="data-inspector">
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
        {draft.prefab && (
          <section className="prefab-override-strip">
            <div><strong>Prefab Instance</strong><span>{draft.prefab.overrideCount} override{draft.prefab.overrideCount === 1 ? '' : 's'}</span></div>
            {draft.prefab.sourceMissing && <b>Source missing</b>}
            <details>
              <summary>Overrides</summary>
              <div className="prefab-override-list">
                {draft.prefab.overrides.map((override) => (
                  <div key={`${override.sourceEntity}:${override.componentId}:${override.fieldId}:${override.kind}`}>
                    <span><b>{override.kind}</b><code>{override.componentId}</code><small>Field {override.fieldId}</small></span>
                    <button onClick={() => void command('prefab.revertOverride', { entity: draft.entity, ...override }).then(refresh)}>Revert</button>
                  </div>
                ))}
                {!draft.prefab.overrides.length && <small>No authored overrides.</small>}
              </div>
            </details>
            <button aria-label="Apply all prefab overrides" onClick={() => void command('prefab.apply', { entity: draft.entity }).then(refresh)}>Apply All</button>
            <button aria-label="Revert all prefab overrides" onClick={() => void command('prefab.revert', { entity: draft.entity }).then(refresh)}>Revert All</button>
            <button aria-label="Unpack prefab from override strip" onClick={() => void command('prefab.unpack', { entity: draft.entity }).then(refresh)}>Unpack</button>
          </section>
        )}
        {schemas.map((schema) => (
          <SchemaComponentCard
            key={schema.id}
            collapsed={collapsed[schema.id] ?? false}
            context={draft}
            schema={schema}
            assets={assets}
            thumbnailProvider={thumbnailProvider}
            onToggle={() => setCollapsed((value) => ({ ...value, [schema.id]: !(value[schema.id] ?? false) }))}
            onAction={(action) => runComponentAction(schema.id, action)}
            onValue={(path, value, settled) => {
              if (path === 'terrain.activeLayer') value = Number(value);
              let next = setPathValue(draft, path, value);
              if (path === 'transform.rotationDegrees' && next.transform) {
                next = { ...next, transform: { ...next.transform, rotationDegrees: value as Vec3 } };
              }
              updateComponent(schema.id, path, next, settled);
            }}
          />
        ))}
        {!schemas.length && <div className="inspector-state compact">No components match “{filter}”.</div>}
        <details className="inspector-add-component">
          <summary>Add Component</summary>
          {[
            ['camera', 'Camera'],
            ['meshRenderer', 'Mesh Renderer'],
            ['directionalLight', 'Directional Light'],
            ['pointLight', 'Point Light'],
            ['spotLight', 'Spot Light'],
            ['areaLight', 'Area Light'],
          ].map(([component, label]) => (
            <button
              key={component}
              onClick={() =>
                void command('component.add', { component }).then(async (response) => {
                  if (!response.succeeded) setError(response.error || `Could not add ${label}`);
                  else await refresh();
                })
              }
              type="button"
            >
              {label}
            </button>
          ))}
          {projectSchemas
            .filter(
              (schema) =>
                schema.projectComponent && !draft.projectComponents.some((component) => component.typeId === schema.id),
            )
            .map((schema) => (
              <button
                key={schema.id}
                onClick={() =>
                  void command('component.add', { component: schema.id }).then(async (response) => {
                    if (!response.succeeded) setError(response.error || `Could not add ${schema.displayName}`);
                    else await refresh();
                  })
                }
                title={schema.tooltip}
                type="button"
              >
                {schema.category ? `${schema.category} / ` : ''}
                {schema.displayName}
              </button>
            ))}
        </details>
      </div>
    </section>
  );
}
