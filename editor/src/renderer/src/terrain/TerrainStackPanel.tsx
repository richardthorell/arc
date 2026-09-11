import {
  ArrowDown,
  ArrowUp,
  Eye,
  EyeOff,
  Layers3,
  Mountain,
  Paintbrush,
  Plus,
  RefreshCw,
  Trash2,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import type { HostEntityId, HostResponse } from '../inspector/inspectorTypes';

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
  modifiers: TerrainModifierSnapshot[];
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

  const execute = useCallback(
    async (operation: string, extra: Record<string, unknown> = {}) => {
      setLoading(true);
      try {
        const response = await command('terrain.modifierStack', { entity, operation, ...extra });
        if (!response.succeeded || !response.payload) {
          onStatus?.(response.error || 'Terrain modifier operation failed');
          return null;
        }
        setStack(response.payload);
        return response.payload;
      } finally {
        setLoading(false);
      }
    },
    [command, entity, onStatus],
  );

  useEffect(() => {
    void execute('inspect');
  }, [execute]);

  useEffect(() => {
    if (!stack) return;
    const current = stack.modifiers.find((modifier) => modifier.id === selectedId);
    if (current) return;
    setSelectedId(stack.modifiers.at(-1)?.id ?? '');
  }, [selectedId, stack]);

  const selected = useMemo(
    () => stack?.modifiers.find((modifier) => modifier.id === selectedId) ?? null,
    [selectedId, stack],
  );

  useEffect(() => {
    setRenameValue(selected?.name ?? '');
  }, [selected]);

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
        <button aria-label="Refresh terrain stack" disabled={loading} onClick={() => void execute('inspect')} type="button">
          <RefreshCw className={loading ? 'spin' : ''} size={14} />
        </button>
      </header>

      {!stack.assetBacked ? (
        <div className="terrain-stack-empty">
          <Mountain size={24} />
          <strong>No TerrainAsset assigned</strong>
          <p>Modifier layers are available for asset-backed terrain. Legacy inline terrain remains editable with the existing brush path.</p>
        </div>
      ) : (
        <>
          <div className="terrain-stack-toolbar">
            <button disabled={stack.readOnly || loading} onClick={() => void add('sculpt')} type="button">
              <Plus size={13} /> <Mountain size={13} /> Sculpt
            </button>
            <button disabled={stack.readOnly || loading} onClick={() => void add('paint')} type="button">
              <Plus size={13} /> <Paintbrush size={13} /> Paint
            </button>
          </div>

          <div className="terrain-stack-list" role="listbox" aria-label="Terrain modifiers">
            {stack.modifiers.map((modifier, index) => {
              const Icon = modifier.type === 'paint' ? Paintbrush : Mountain;
              return (
                <div
                  aria-selected={modifier.id === selectedId}
                  className={`terrain-stack-row${modifier.id === selectedId ? ' selected' : ''}`}
                  key={modifier.id}
                  onClick={() => setSelectedId(modifier.id)}
                  role="option"
                  tabIndex={0}
                >
                  <button
                    aria-label={`${modifier.enabled ? 'Disable' : 'Enable'} ${modifier.name}`}
                    disabled={stack.readOnly || loading}
                    onClick={(event) => {
                      event.stopPropagation();
                      void mutate('set_enabled', { modifier: modifier.id, enabled: !modifier.enabled });
                    }}
                    type="button"
                  >
                    {modifier.enabled ? <Eye size={14} /> : <EyeOff size={14} />}
                  </button>
                  <Icon size={14} />
                  <span className="terrain-stack-row-copy">
                    <strong>{modifier.name}</strong>
                    <small>
                      {modifier.type === 'paint' ? 'Paint Layer' : modifier.type === 'sculpt' ? 'Sculpt Layer' : modifier.typeId}
                      {modifier.regionPayloads > 0 ? ` · ${modifier.regionPayloads} region${modifier.regionPayloads === 1 ? '' : 's'}` : ''}
                    </small>
                  </span>
                  <span className="terrain-stack-row-actions">
                    <button
                      aria-label={`Move ${modifier.name} up`}
                      disabled={stack.readOnly || loading || index === 0}
                      onClick={(event) => {
                        event.stopPropagation();
                        void mutate('move', { modifier: modifier.id, index: index - 1 });
                      }}
                      type="button"
                    >
                      <ArrowUp size={13} />
                    </button>
                    <button
                      aria-label={`Move ${modifier.name} down`}
                      disabled={stack.readOnly || loading || index === stack.modifiers.length - 1}
                      onClick={(event) => {
                        event.stopPropagation();
                        void mutate('move', { modifier: modifier.id, index: index + 1 });
                      }}
                      type="button"
                    >
                      <ArrowDown size={13} />
                    </button>
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
            <div className="terrain-stack-properties">
              <h3>Selected Modifier</h3>
              <label>
                <span>Name</span>
                <input
                  aria-label="Modifier name"
                  disabled={stack.readOnly || loading}
                  onBlur={() => void commitRename()}
                  onChange={(event) => setRenameValue(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') void commitRename();
                  }}
                  value={renameValue}
                />
              </label>
              <div className="terrain-stack-property">
                <span>Type</span>
                <strong>{selected.type === 'paint' ? 'Paint Layer' : selected.type === 'sculpt' ? 'Sculpt Layer' : selected.typeId}</strong>
              </div>
              <div className="terrain-stack-property">
                <span>Stable ID</span>
                <code title={selected.id}>{selected.id.slice(0, 12)}…</code>
              </div>
              <button
                className="terrain-stack-delete"
                disabled={stack.readOnly || loading}
                onClick={() => void mutate('erase', { modifier: selected.id })}
                type="button"
              >
                <Trash2 size={13} /> Delete Modifier
              </button>
            </div>
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
