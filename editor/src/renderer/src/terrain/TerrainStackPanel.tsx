import {
  ArrowDown,
  ArrowUp,
  Copy,
  Eye,
  EyeOff,
  GripVertical,
  Layers3,
  Mountain,
  Paintbrush,
  Plus,
  RefreshCw,
  Trash2,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import type { HostEntityId, HostResponse } from '../inspector/inspectorTypes';
import { UiButton, UiIconButton, UiPropertyCard, UiTextInput } from '../ui';

import './terrainEditor.css';

export type TerrainModifierSnapshot = {
  id: string;
  name: string;
  type: 'sculpt' | 'paint' | 'unknown';
  typeId: string;
  enabled: boolean;
  regionPayloads: number;
};

export type TerrainModifierStackSnapshot = {
  entity: HostEntityId;
  assetBacked: boolean;
  readOnly: boolean;
  assetPath: string;
  authoringRevision: number;
  activeModifier: string;
  modifiers: TerrainModifierSnapshot[];
  rebuild: {
    state: 'idle' | 'queued' | 'building' | 'publishing' | 'failed';
    authoringRevision: number;
    dirtyRegions: number;
    geometryRegions: number;
    attributeRegions: number;
    error: string;
  };
};

type TerrainStackPanelProps = {
  entity: HostEntityId;
  command: (type: string, payload: unknown) => Promise<HostResponse<TerrainModifierStackSnapshot>>;
  onStatus?: (message: string) => void;
};

export function TerrainStackPanel({ entity, command, onStatus }: TerrainStackPanelProps) {
  const [stack, setStack] = useState<TerrainModifierStackSnapshot | null>(null);
  const [selectedId, setSelectedId] = useState<string>('');
  const [renameValue, setRenameValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [draggedId, setDraggedId] = useState('');
  const [requestError, setRequestError] = useState('');

  const execute = useCallback(
    async (operation: string, extra: Record<string, unknown> = {}, silent = false) => {
      if (!silent) setLoading(true);
      try {
        const response = await command('terrain.modifierStack', { entity, operation, ...extra });
        if (!response.succeeded || !response.payload) {
          const message = response.error || 'Terrain modifier operation failed';
          setRequestError(message);
          onStatus?.(message);
          return null;
        }
        setRequestError('');
        setStack(response.payload);
        return response.payload;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Terrain modifier operation failed';
        setRequestError(message);
        onStatus?.(message);
        return null;
      } finally {
        if (!silent) setLoading(false);
      }
    },
    [command, entity, onStatus],
  );

  useEffect(() => {
    void execute('inspect');
  }, [execute]);

  useEffect(() => {
    if (!stack || stack.rebuild.state === 'idle' || stack.rebuild.state === 'failed') return;
    const timer = window.setTimeout(() => void execute('inspect', {}, true), 250);
    return () => window.clearTimeout(timer);
  }, [execute, stack]);

  useEffect(() => {
    if (!stack) return;
    const current = stack.modifiers.find((modifier) => modifier.id === selectedId);
    if (current) return;
    const active = stack.modifiers.find((modifier) => modifier.id === stack.activeModifier);
    setSelectedId(active?.id ?? stack.modifiers.at(-1)?.id ?? '');
  }, [selectedId, stack]);

  const selected = useMemo(
    () => stack?.modifiers.find((modifier) => modifier.id === selectedId) ?? null,
    [selectedId, stack],
  );

  useEffect(() => {
    setRenameValue(selected?.name ?? '');
  }, [selected]);

  const selectModifier = async (modifier: TerrainModifierSnapshot) => {
    setSelectedId(modifier.id);
    const next = await execute('select', { modifier: modifier.id });
    if (!next) return;
    setSelectedId(modifier.id);
  };

  const mutate = async (operation: string, extra: Record<string, unknown> = {}) => {
    const next = await execute(operation, extra);
    if (next) onStatus?.(`Terrain stack ${operation.replaceAll('_', ' ')} completed`);
    return next;
  };

  const add = async (kind: 'sculpt' | 'paint') => {
    const next = await mutate(kind === 'sculpt' ? 'add_sculpt' : 'add_paint');
    if (next?.modifiers.length) setSelectedId(next.modifiers.at(-1)?.id ?? '');
  };

  const commitRename = async () => {
    if (!selected || renameValue.trim() === selected.name || !renameValue.trim()) return;
    await mutate('rename', { modifier: selected.id, name: renameValue.trim() });
  };

  if (!stack) {
    return (
      <section className="terrain-stack-panel" aria-label="Terrain stack">
        <div className="terrain-stack-loading">
          <RefreshCw className={loading ? 'spin' : ''} size={16} /> Loading terrain stack…
        </div>
        {requestError && <small role="alert">{requestError}</small>}
      </section>
    );
  }

  return (
    <section className="terrain-stack-panel" aria-label="Terrain stack">
      <header className="terrain-stack-header">
        <div>
          <span className="terrain-stack-icon">
            <Layers3 size={17} />
          </span>
          <span>
            <strong>Terrain Stack</strong>
            <small>{stack.assetBacked ? stack.assetPath || 'Terrain Asset' : 'Legacy inline terrain'}</small>
          </span>
        </div>
        <UiIconButton
          disabled={loading}
          label="Refresh terrain stack"
          onClick={() => void execute('inspect')}
          type="button"
        >
          <RefreshCw className={loading ? 'spin' : ''} size={14} />
        </UiIconButton>
      </header>

      {!stack.assetBacked ? (
        <div className="terrain-stack-empty">
          <Mountain size={24} />
          <strong>No TerrainAsset assigned</strong>
          <p>
            Modifier layers are available for asset-backed terrain. Legacy inline terrain remains editable with the
            existing brush path.
          </p>
        </div>
      ) : (
        <>
          <div
            aria-live="polite"
            className={`terrain-rebuild-status ${stack.rebuild.state}`}
            role={stack.rebuild.state === 'failed' ? 'alert' : 'status'}
          >
            <RefreshCw
              className={['queued', 'building', 'publishing'].includes(stack.rebuild.state) ? 'spin' : ''}
              size={14}
            />
            <span>
              <strong>
                {stack.rebuild.state === 'idle'
                  ? 'Terrain is up to date'
                  : stack.rebuild.state === 'failed'
                    ? 'Terrain rebuild failed'
                    : `${stack.rebuild.state[0].toUpperCase()}${stack.rebuild.state.slice(1)} ${stack.rebuild.dirtyRegions} region${stack.rebuild.dirtyRegions === 1 ? '' : 's'}`}
              </strong>
              {stack.rebuild.dirtyRegions > 0 && (
                <small>
                  {stack.rebuild.geometryRegions} geometry · {stack.rebuild.attributeRegions} attribute
                </small>
              )}
              {(stack.rebuild.error || requestError) && <small>{stack.rebuild.error || requestError}</small>}
            </span>
          </div>
          <div className="terrain-stack-toolbar">
            <UiButton disabled={stack.readOnly || loading} onClick={() => void add('sculpt')} type="button">
              <Plus size={13} /> <Mountain size={13} /> Sculpt
            </UiButton>
            <UiButton disabled={stack.readOnly || loading} onClick={() => void add('paint')} type="button">
              <Plus size={13} /> <Paintbrush size={13} /> Paint
            </UiButton>
          </div>

          <div className="terrain-stack-list" role="listbox" aria-label="Terrain modifiers">
            {stack.modifiers.map((modifier, index) => {
              const Icon = modifier.type === 'paint' ? Paintbrush : Mountain;
              return (
                <div
                  aria-selected={modifier.id === selectedId}
                  className={`terrain-stack-row${modifier.id === selectedId ? ' selected' : ''}${draggedId === modifier.id ? ' dragging' : ''}`}
                  draggable={!stack.readOnly && !loading}
                  key={modifier.id}
                  onClick={() => void selectModifier(modifier)}
                  onDragEnd={() => setDraggedId('')}
                  onDragOver={(event) => {
                    if (draggedId && draggedId !== modifier.id) event.preventDefault();
                  }}
                  onDragStart={(event) => {
                    setDraggedId(modifier.id);
                    event.dataTransfer.effectAllowed = 'move';
                    event.dataTransfer.setData('text/plain', modifier.id);
                  }}
                  onDrop={(event) => {
                    event.preventDefault();
                    const source = draggedId || event.dataTransfer.getData('text/plain');
                    setDraggedId('');
                    if (source && source !== modifier.id) void mutate('move', { modifier: source, index });
                  }}
                  role="option"
                  tabIndex={0}
                >
                  <GripVertical aria-hidden="true" className="terrain-stack-drag-handle" size={13} />
                  <UiIconButton
                    disabled={stack.readOnly || loading}
                    label={`${modifier.enabled ? 'Disable' : 'Enable'} ${modifier.name}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      void mutate('set_enabled', { modifier: modifier.id, enabled: !modifier.enabled });
                    }}
                    type="button"
                  >
                    {modifier.enabled ? <Eye size={14} /> : <EyeOff size={14} />}
                  </UiIconButton>
                  <Icon size={14} />
                  <span className="terrain-stack-row-copy">
                    <strong>{modifier.name}</strong>
                    <small>
                      {modifier.type === 'paint'
                        ? 'Paint Layer'
                        : modifier.type === 'sculpt'
                          ? 'Sculpt Layer'
                          : modifier.typeId}
                      {modifier.regionPayloads > 0
                        ? ` · ${modifier.regionPayloads} region${modifier.regionPayloads === 1 ? '' : 's'}`
                        : ''}
                    </small>
                  </span>
                  <span className="terrain-stack-row-actions">
                    <UiIconButton
                      disabled={stack.readOnly || loading || index === 0}
                      label={`Move ${modifier.name} up`}
                      onClick={(event) => {
                        event.stopPropagation();
                        void mutate('move', { modifier: modifier.id, index: index - 1 });
                      }}
                      type="button"
                    >
                      <ArrowUp size={13} />
                    </UiIconButton>
                    <UiIconButton
                      disabled={stack.readOnly || loading || index === stack.modifiers.length - 1}
                      label={`Move ${modifier.name} down`}
                      onClick={(event) => {
                        event.stopPropagation();
                        void mutate('move', { modifier: modifier.id, index: index + 1 });
                      }}
                      type="button"
                    >
                      <ArrowDown size={13} />
                    </UiIconButton>
                  </span>
                </div>
              );
            })}
            <div className="terrain-stack-row base" aria-disabled="true">
              <span className="terrain-stack-lock">◆</span>
              <Mountain size={14} />
              <span className="terrain-stack-row-copy">
                <strong>Base Source</strong>
                <small>Immutable terrain source</small>
              </span>
            </div>
          </div>

          {selected && (
            <UiPropertyCard
              className="terrain-stack-properties"
              expandable={false}
              fields={[
                {
                  id: 'name',
                  label: 'Name',
                  control: (
                    <UiTextInput
                      aria-label="Modifier name"
                      disabled={stack.readOnly || loading}
                      onBlur={() => void commitRename()}
                      onChange={(event) => setRenameValue(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') void commitRename();
                      }}
                      value={renameValue}
                    />
                  ),
                },
                {
                  id: 'type',
                  label: 'Type',
                  control: (
                    <strong>
                      {selected.type === 'paint'
                        ? 'Paint Layer'
                        : selected.type === 'sculpt'
                          ? 'Sculpt Layer'
                          : selected.typeId}
                    </strong>
                  ),
                },
                {
                  id: 'stable-id',
                  label: 'Stable ID',
                  control: <code title={selected.id}>{selected.id.slice(0, 12)}…</code>,
                },
                {
                  id: 'duplicate',
                  fullWidth: true,
                  control: (
                    <UiButton
                      disabled={stack.readOnly || loading}
                      onClick={async () => {
                        const next = await mutate('duplicate', { modifier: selected.id });
                        if (next?.activeModifier) setSelectedId(next.activeModifier);
                      }}
                      type="button"
                    >
                      <Copy size={13} /> Duplicate Modifier
                    </UiButton>
                  ),
                },
                {
                  id: 'delete',
                  fullWidth: true,
                  control: (
                    <UiButton
                      className="terrain-stack-delete"
                      disabled={stack.readOnly || loading}
                      onClick={() => void mutate('erase', { modifier: selected.id })}
                      type="button"
                      variant="danger"
                    >
                      <Trash2 size={13} /> Delete Modifier
                    </UiButton>
                  ),
                },
              ]}
              title="Selected Modifier"
            />
          )}

          <footer className="terrain-stack-footer">
            Revision {stack.authoringRevision}
            {stack.readOnly && <span>Read only</span>}
          </footer>
        </>
      )}
    </section>
  );
}
