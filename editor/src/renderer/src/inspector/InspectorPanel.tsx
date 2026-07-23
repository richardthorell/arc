import { useEffect, useMemo, useRef, useState } from 'react';
import { Filter, MoreVertical, Search } from 'lucide-react';

import type { AssetPickerItem, AssetThumbnailProvider } from './AssetPicker';

import {
  schemaForSnapshot,
  setPathValue,
} from './componentSchemas';
import type { InspectorComponentId } from './componentSchemas';
import type { HostResponse, InspectorEntitySnapshot, Vec3 } from './inspectorTypes';
import { cameraHostPayload, transformHostPayload } from './inspectorTypes';
import { SchemaComponentCard } from './SchemaComponents';

import './inspector.css';

export type InspectorEditTransaction = { id: number; phase: 'begin' | 'update' | 'commit' | 'cancel'; label?: string };
export type InspectorCommand = (type: string, payload: Record<string, unknown>, edit?: InspectorEditTransaction) => Promise<HostResponse>;

export type InspectorPanelProps = {
  snapshot: InspectorEntitySnapshot | null;
  loading?: boolean;
  command: InspectorCommand;
  refresh: () => Promise<void>;
  onStatus?: (message: string) => void;
  assets?: ReadonlyArray<AssetPickerItem>;
  thumbnailProvider?: AssetThumbnailProvider;
};

const knownTags = ['Untagged', 'Camera', 'Light', 'Mesh', 'Environment'];
const defaultLayerMask = 1;
const environmentLayerMask = 2;

function entityPayload(snapshot: InspectorEntitySnapshot) {
  return { entity: snapshot.entity };
}

function TextCommitInput({ ariaLabel, value, onCommit, list }: {
  ariaLabel: string;
  value: string;
  onCommit: (value: string) => void;
  list?: string;
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
      value={draft}
    />
  );
}

export function InspectorPanel({ snapshot, loading, command, refresh, onStatus, assets = [], thumbnailProvider }: InspectorPanelProps) {
  const [draft, setDraft] = useState(snapshot);
  const [filter, setFilter] = useState('');
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const confirmed = useRef(snapshot);
  const revision = useRef(0);
  const nextTransactionId = useRef(1);
  const activeTransaction = useRef<{ id: number; key: string } | null>(null);

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

  const updateComponent = (component: InspectorComponentId, path: string, next: InspectorEntitySnapshot, settled: boolean) => {
    const transactionKey = `${component}:${path}`;
    const transactionLabel = component === 'transform' ? 'Transform Entity' : component === 'camera' ? 'Edit Camera' :
      component === 'meshRenderer' ? 'Edit Mesh Renderer' : 'Edit Terrain';
    if (component === 'transform' && next.transform) {
      void runMutation(next, 'entity.setTransform', {
        ...entityPayload(next), transform: transformHostPayload(next.transform),
      }, settled, transactionKey, transactionLabel);
    } else if (component === 'camera' && next.camera) {
      void runMutation(next, 'entity.setCamera', {
        ...entityPayload(next), camera: cameraHostPayload(next.camera),
      }, settled, transactionKey, transactionLabel);
    } else if (component === 'meshRenderer' && next.meshRenderer) {
      if (path === 'meshRenderer.materialPath') {
        void runMutation(next, 'entity.setMaterial', {
          ...entityPayload(next), path: next.meshRenderer.materialPath,
        }, true);
      } else {
        const tint = next.meshRenderer.baseColorTint;
        void runMutation(next, 'entity.setMeshRenderer', {
          ...entityPayload(next),
          visible: next.meshRenderer.visible,
          baseColorTint: [tint.x, tint.y, tint.z, tint.w],
        }, settled, transactionKey, transactionLabel);
      }
    } else if (component === 'terrain' && next.terrain) {
      const layerMatch = /^terrain\.layers\.(\d)\.baseColorPath$/.exec(path);
      if (layerMatch) {
        void runMutation(next, 'terrain.assignLayer', {
          ...entityPayload(next), layer: Number(layerMatch[1]), path: next.terrain.layers[Number(layerMatch[1])].baseColorPath,
        }, true);
      } else {
        void runMutation(next, 'terrain.update', {
          ...entityPayload(next), enabled: next.terrain.enabled, receiveShadows: next.terrain.receiveShadows,
        }, settled, transactionKey, transactionLabel);
      }
    } else if (component === 'terrainBrush' && next.terrain) {
      void runMutation(next, 'terrain.setBrush', {
        ...entityPayload(next), tool: next.terrain.brushTool, radius: next.terrain.brushRadius,
        strength: next.terrain.brushStrength, falloff: next.terrain.brushFalloff,
        activeLayer: Number(next.terrain.activeLayer),
      }, true);
    }
  };

  const schemas = useMemo(() => {
    if (!draft) return [];
    const needle = filter.trim().toLocaleLowerCase();
    return schemaForSnapshot(draft).filter((schema) => !needle ||
      schema.title.toLocaleLowerCase().includes(needle) ||
      schema.fields.some((field) => field.label.toLocaleLowerCase().includes(needle)));
  }, [draft, filter]);

  if (loading) return <div className="inspector-state">Loading selection…</div>;
  if (!draft) return <div className="inspector-state">Select an entity to inspect its components.</div>;

  const layerValue = draft.renderLayerMask === defaultLayerMask
    ? String(defaultLayerMask)
    : draft.renderLayerMask === environmentLayerMask ? String(environmentLayerMask) : `custom:${draft.renderLayerMask}`;
  const tagOptions = knownTags.includes(draft.tag || 'Untagged')
    ? knownTags
    : [...knownTags, draft.tag];

  return (
    <section className="data-inspector">
      <header className="inspector-entity-card">
        <div className="inspector-entity-title-row">
          <input
            aria-label="Entity active"
            checked={draft.active}
            onChange={(event) => updateHeader({ ...draft, active: event.target.checked }, 'entity.setActive', { active: event.target.checked })}
            type="checkbox"
          />
          <TextCommitInput
            ariaLabel="Entity name"
            value={draft.name}
            onCommit={(name) => updateHeader({ ...draft, name }, 'entity.rename', { name })}
          />
          <label className="inspector-static" title="Static mobility will be available when ARC adds an ECS mobility contract.">
            <input aria-label="Static" disabled type="checkbox" />
            <span>Static</span>
          </label>
          <button aria-label="Entity actions" className="inspector-menu-button" type="button"><MoreVertical size={15} /></button>
        </div>
        <div className="inspector-entity-meta-row">
          <label><span>Tag</span>
            <TextCommitInput
              ariaLabel="Tag"
              list="arc-inspector-tags"
              value={draft.tag || 'Untagged'}
              onCommit={(value) => {
                const tag = value === 'Untagged' ? '' : value;
                updateHeader({ ...draft, tag }, 'entity.setTag', { tag });
              }}
            />
            <datalist id="arc-inspector-tags">
              {tagOptions.map((tag) => <option key={tag} value={tag} />)}
            </datalist>
          </label>
          <label><span>Layer</span>
            <select
              aria-label="Layer"
              value={layerValue}
              onChange={(event) => {
                if (event.target.value.startsWith('custom:')) return;
                const renderLayerMask = Number(event.target.value);
                updateHeader({ ...draft, renderLayerMask }, 'entity.setRenderLayer', { renderLayerMask });
              }}
            >
              <option value={String(defaultLayerMask)}>Default</option>
              <option value={String(environmentLayerMask)}>Environment</option>
              {layerValue.startsWith('custom:') && <option value={layerValue}>{`Custom (0x${draft.renderLayerMask.toString(16).toUpperCase()})`}</option>}
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
        <button aria-label="Component filter options" type="button"><Filter size={17} /></button>
      </div>

      {error && <div className="inspector-error" role="alert">{error}</div>}
      <div className="inspector-component-list">
        {schemas.map((schema) => (
          <SchemaComponentCard
            key={schema.id}
            collapsed={collapsed[schema.id] ?? false}
            context={draft}
            schema={schema}
            assets={assets}
            thumbnailProvider={thumbnailProvider}
            onToggle={() => setCollapsed((value) => ({ ...value, [schema.id]: !(value[schema.id] ?? false) }))}
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
      </div>
    </section>
  );
}
