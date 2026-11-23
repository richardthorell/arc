import { useCallback, useEffect, useRef, useState } from 'react';
import type { LivePlayEntity, LivePlayMessage } from '../../../common/livePlayProtocol';
import type { WindowsLivePlaySnapshot } from '../../../common/livePlaySessionTypes';
import {
  chooseRuntimeInspection,
  matchesRuntimeEntity,
  runtimeEntityKey,
  runtimeHierarchyRows,
  type RuntimeHierarchyMessage,
  type RuntimeInspectionResult,
} from './runtimeInspectionModel';
import './runtimeInspection.css';

const inspectableComponents = [
  { typeId: 'arc.transform', title: 'Transform' },
  { typeId: 'arc.scene.name', title: 'Name' },
  { typeId: 'arc.scene.tag', title: 'Tag' },
  { typeId: 'arc.scene.active', title: 'Active' },
] as const;

type InspectionView = {
  scope: string;
  hierarchy: RuntimeHierarchyMessage | null;
  selected: LivePlayEntity | null;
  properties: Record<string, RuntimeInspectionResult>;
  notice: string | null;
  requiresSnapshot: boolean;
};

const emptyView = (scope: string): InspectionView => ({
  scope,
  hierarchy: null,
  selected: null,
  properties: {},
  notice: null,
  requiresSnapshot: false,
});

function sameSession(message: LivePlayMessage, snapshot: WindowsLivePlaySnapshot): boolean {
  return Boolean(
    snapshot.sessionId &&
      snapshot.instanceId &&
      message.sessionId === snapshot.sessionId &&
      message.instanceId === snapshot.instanceId,
  );
}

/**
 * External Development process state is independent of the native editor-host
 * Play World and its Authoring World selection.
 */
export function useLivePlayRuntimeInspection(snapshot: WindowsLivePlaySnapshot) {
  const scope = snapshot.sessionId && snapshot.instanceId ? `${snapshot.sessionId}:${snapshot.instanceId}` : '';
  const [view, setView] = useState<InspectionView>(() => emptyView(scope));
  const pending = useRef(new Map<string, { scope: string; entity: string; component: string }>());
  const current = view.scope === scope ? view : emptyView(scope);

  useEffect(() => {
    if (!scope || !window.arc?.livePlay) return;
    let subscribed = true;
    pending.current.clear();
    const onMessage = (message: LivePlayMessage) => {
      if (!sameSession(message, snapshot)) return;
      if (message.type === 'component.inspect.result' && message.payload.requestId) {
        const sent = pending.current.get(message.payload.requestId);
        if (sent) {
          pending.current.delete(message.payload.requestId);
          if (
            sent.scope !== scope ||
            sent.entity !== runtimeEntityKey(message.payload.entity) ||
            sent.component !== message.payload.component.typeId
          )
            return;
        }
      }
      if (message.type === 'command.ack') {
        const sent = pending.current.get(message.payload.requestId);
        if (!sent) return;
        pending.current.delete(message.payload.requestId);
        if (message.payload.status === 'rejected' && message.payload.error.code !== 'unsupported_operation') {
          const errorMessage = message.payload.error.message;
          setView((previous) => (previous.scope === scope ? { ...previous, notice: errorMessage } : previous));
        }
        return;
      }
      setView((previous) => {
        if (previous.scope !== scope) previous = emptyView(scope);
        if (message.type === 'hierarchy.snapshot') {
          if (previous.hierarchy && message.payload.worldRevision < previous.hierarchy.payload.worldRevision)
            return previous;
          const selected = previous.selected;
          const present =
            selected &&
            message.payload.entities.some((item) => matchesRuntimeEntity(selected, item.entity));
          const keepSelection = present || !message.payload.complete;
          return {
            ...previous,
            hierarchy: message,
            requiresSnapshot: false,
            selected: keepSelection ? selected : null,
            properties: keepSelection ? previous.properties : {},
            notice: null,
          };
        }
        // Deltas require a complete, gap-free base; L1.5c adds a resync API.
        // Never display a potentially incorrect tree after seeing a delta.
        if (message.type === 'hierarchy.changed')
          return { ...previous, requiresSnapshot: true, notice: 'Runtime hierarchy changed; full resync required.' };
        if (
          message.type === 'component.inspect.result' &&
          matchesRuntimeEntity(previous.selected, message.payload.entity)
        ) {
          const typeId = message.payload.component.typeId;
          return {
            ...previous,
            properties: {
              ...previous.properties,
              [typeId]: chooseRuntimeInspection(previous.properties[typeId], message.payload),
            },
          };
        }
        return previous;
      });
    };
    const unsubscribe = window.arc.livePlay.onMessage(onMessage);
    void window.arc.livePlay
      .latestHierarchy()
      .then((message) => {
        if (subscribed && message && sameSession(message, snapshot)) onMessage(message);
      })
      .catch(() => {});
    return () => {
      subscribed = false;
      pending.current.clear();
      unsubscribe();
    };
    // Scope is the complete session/instance identity. State changes do not reset the hierarchy.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope]);

  const select = useCallback(
    (entity: LivePlayEntity) => {
      pending.current.clear();
      setView((previous) => ({
        ...(previous.scope === scope ? previous : emptyView(scope)),
        selected: entity,
        properties: {},
        notice: null,
      }));
    },
    [scope],
  );

  const selected = current.selected;
  const requestSelected = useCallback(() => {
    if (!selected || !scope || snapshot.state !== 'connected') return;
    const entityKey = runtimeEntityKey(selected);
    for (const component of inspectableComponents) {
      void window.arc.livePlay
        .inspect(selected, component.typeId)
        .then((requestId) => {
          if (!requestId) return;
          if (pending.current.size >= 64) pending.current.clear();
          pending.current.set(requestId, { scope, entity: entityKey, component: component.typeId });
        })
        .catch(() => {
          setView((previous) =>
            previous.scope === scope ? { ...previous, notice: 'Native runtime inspection is unavailable.' } : previous,
          );
        });
    }
  }, [scope, selected, snapshot.state]);

  useEffect(() => {
    if (!selected || snapshot.state !== 'connected') return;
    requestSelected();
    const interval = window.setInterval(requestSelected, 1200);
    return () => window.clearInterval(interval);
  }, [selected, snapshot.state, requestSelected]);

  return {
    hierarchy: current.hierarchy,
    selected,
    properties: current.properties,
    notice: current.notice,
    requiresSnapshot: current.requiresSnapshot,
    status: snapshot.state,
    select,
    refresh: requestSelected,
  };
}

export type LivePlayRuntimeInspection = ReturnType<typeof useLivePlayRuntimeInspection>;

export function RuntimeHierarchyPanel({ inspection }: { inspection: LivePlayRuntimeInspection }) {
  const hierarchy = inspection.hierarchy?.payload;
  const rows = hierarchy ? runtimeHierarchyRows(hierarchy.entities) : [];
  const disconnected = inspection.status !== 'connected';
  return (
    <section className="live-runtime-panel" aria-label="External runtime hierarchy">
      <header className="live-runtime-panel-heading">
        <strong>Runtime Hierarchy</strong>
        <span>External Development game · read-only</span>
      </header>
      {disconnected && (
        <p className="live-runtime-warning" role="status">
          {inspection.status === 'disconnected'
            ? 'Disconnected — last received runtime hierarchy is stale.'
            : `Runtime ${inspection.status}. Inspection is unavailable.`}
        </p>
      )}
      {inspection.requiresSnapshot && (
        <p className="live-runtime-warning" role="status">
          Full hierarchy resync required; changes are not displayed until a new snapshot arrives.
        </p>
      )}
      {hierarchy && !hierarchy.complete && (
        <p className="live-runtime-warning" role="status">
          Partial hierarchy — the native protocol currently limits snapshots to 128 entities.
        </p>
      )}
      <div className="live-runtime-entity-list" role="tree" aria-label="Running game entities">
        {rows.map((row) => (
          <button
            type="button"
            role="treeitem"
            key={row.key}
            aria-selected={matchesRuntimeEntity(inspection.selected, row.entity)}
            className={matchesRuntimeEntity(inspection.selected, row.entity) ? 'selected' : ''}
            style={{ paddingLeft: `${12 + row.depth * 17}px` }}
            onClick={() => inspection.select(row.entity)}
          >
            <span className="live-runtime-entity-name">{row.name || 'Unnamed entity'}</span>
            <small>{row.entity.kind === 'authored' ? 'Authored' : 'Runtime'}</small>
          </button>
        ))}
      </div>
      {rows.length === 0 && (
        <p className="live-runtime-empty">
          {disconnected ? 'No cached runtime hierarchy.' : 'Waiting for a native runtime hierarchy snapshot…'}
        </p>
      )}
      <footer className="live-runtime-footer">
        {rows.length} visible entities
        {hierarchy ? ` · Revision ${hierarchy.worldRevision}` : ''}
      </footer>
    </section>
  );
}

function valueText(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((item) => (typeof item === 'number' ? Number(item.toFixed(4)) : item)).join(', ');
  }
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value === 'string') return value;
  return 'Unavailable';
}

export function RuntimeInspectorPanel({ inspection }: { inspection: LivePlayRuntimeInspection }) {
  const selected = inspection.selected;
  const rows = inspection.hierarchy?.payload.entities ?? [];
  const matching = rows.find((item) => selected && matchesRuntimeEntity(selected, item.entity));
  const disconnected = inspection.status !== 'connected';
  return (
    <section className="live-runtime-panel live-runtime-inspector" aria-label="External runtime inspector">
      <header className="live-runtime-panel-heading">
        <strong>Runtime Inspector</strong>
        <span>Actual native game values · never writes to Authoring World</span>
      </header>
      {disconnected && (
        <p className="live-runtime-warning" role="status">
          {inspection.status === 'disconnected'
            ? 'Disconnected — values are stale.'
            : 'Runtime inspection unavailable.'}
        </p>
      )}
      {inspection.notice && <p className="live-runtime-warning" role="status">{inspection.notice}</p>}
      {!selected ? (
        <p className="live-runtime-empty">Select an entity in Runtime Hierarchy to inspect the running game.</p>
      ) : (
        <>
          <div className="live-runtime-identity">
            <strong>{matching?.name || 'Runtime entity'}</strong>
            <small>{selected.kind === 'authored' ? 'Authored GUID' : 'Session-local runtime ID'}</small>
            <code>{selected.kind === 'authored' ? selected.entityGuid : selected.runtimeEntityId}</code>
          </div>
          <div className="live-runtime-inspector-actions">
            <span>Read-only native components</span>
            <button type="button" disabled={disconnected} onClick={inspection.refresh}>
              Refresh
            </button>
          </div>
          {inspectableComponents.map(({ typeId, title }) => {
            const result = inspection.properties[typeId];
            if (!result) return null;
            return (
              <section className="live-runtime-component" key={typeId}>
                <h4>
                  {title} <small>Rev {result.runtimeRevision}</small>
                </h4>
                <dl>
                  {result.properties.map((property) => (
                    <div key={property.path}>
                      <dt>{property.path}</dt>
                      <dd>{valueText(property.value)}</dd>
                    </div>
                  ))}
                </dl>
              </section>
            );
          })}
          {Object.keys(inspection.properties).length === 0 && (
            <p className="live-runtime-empty">
              {disconnected ? 'No values available for the disconnected game.' : 'Reading native components…'}
            </p>
          )}
        </>
      )}
    </section>
  );
}
