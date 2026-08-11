import { useEffect, useMemo, useState } from 'react';
import { Lightbulb, RefreshCw, Search, Sun } from 'lucide-react';

import type { SceneEntity } from '../services/editorHostTypes';
import type { HostResponse } from '../inspector/inspectorTypes';
import { UiIconButton } from '../ui';
import type { EditorDiagnosticsSnapshot } from '../tools/toolTypes';

import '../tools/tools.css';

export function LightingPanel({
  entities,
  onSelect,
}: {
  entities: SceneEntity[];
  onSelect: (entityId: string) => void;
}) {
  const [diagnostics, setDiagnostics] = useState<EditorDiagnosticsSnapshot | null>(null);
  const [filter, setFilter] = useState('');
  const lights = useMemo(() => {
    const query = filter.trim().toLocaleLowerCase();
    const visit = (source: SceneEntity[]): SceneEntity[] =>
      source.flatMap((entity) => [
        ...(entity.kind === 'light' && (!query || entity.name.toLocaleLowerCase().includes(query)) ? [entity] : []),
        ...visit(entity.children ?? []),
      ]);
    return visit(entities);
  }, [entities, filter]);

  const refresh = async () => {
    const response = (await window.arc.host.query('gateway.diagnostics')) as HostResponse<EditorDiagnosticsSnapshot>;
    if (response.succeeded && response.payload) setDiagnostics(response.payload);
  };
  useEffect(() => {
    void refresh();
  }, []);

  return (
    <section className="production-tool-panel lighting-production-panel">
      <header className="tool-panel-toolbar">
        <Lightbulb size={15} />
        <strong>Lighting</strong>
        <UiIconButton label="Refresh lighting diagnostics" onClick={() => void refresh()}>
          <RefreshCw size={14} />
        </UiIconButton>
      </header>
      <label className="tool-search">
        <Search size={14} />
        <input
          aria-label="Search lights"
          placeholder="Search scene lights…"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        />
      </label>
      <section className="lighting-section">
        <h3>Scene lights</h3>
        {lights.map((light) => (
          <button
            className="lighting-entity"
            key={light.guid ?? light.id}
            onClick={() => onSelect(light.id)}
            type="button"
          >
            <Sun size={14} />
            <span>{light.name}</span>
            <small>{light.active ? 'Enabled' : 'Disabled'}</small>
          </button>
        ))}
        {!lights.length && <div className="tool-empty">No matching lights in the loaded scenes.</div>}
      </section>
      <section className="lighting-section">
        <h3>Environment lighting</h3>
        <dl className="tool-property-grid">
          <dt>Source</dt>
          <dd>{diagnostics?.environment.source || 'Unavailable'}</dd>
          <dt>Quality</dt>
          <dd>{diagnostics?.environment.qualityPath || 'Unavailable'}</dd>
          <dt>Atmosphere LUT</dt>
          <dd>{diagnostics?.environment.atmosphereLutState || 'Unavailable'}</dd>
          <dt>IBL</dt>
          <dd>{diagnostics?.environment.lightingState || 'Unavailable'}</dd>
        </dl>
      </section>
      <section className="lighting-section">
        <h3>Shadows</h3>
        <dl className="tool-property-grid">
          <dt>Directional</dt>
          <dd>
            {diagnostics?.shadows.cascades ?? 0} × {diagnostics?.shadows.directionalResolution ?? 0}
          </dd>
          <dt>Local atlas</dt>
          <dd>
            {diagnostics?.shadows.localAtlasResolution ?? 0}px · {diagnostics?.shadows.localAllocations ?? 0}{' '}
            allocations
          </dd>
          <dt>Cache</dt>
          <dd>
            {diagnostics?.shadows.localCacheHits ?? 0} hits / {diagnostics?.shadows.localCacheMisses ?? 0} misses
          </dd>
          <dt>Screen space</dt>
          <dd>{diagnostics?.shadows.screenSpace ? 'Enabled' : 'Disabled'}</dd>
        </dl>
      </section>
      <section className="lighting-section">
        <h3>Global illumination and reflections</h3>
        <dl className="tool-property-grid">
          <dt>Trace path</dt>
          <dd>{diagnostics?.indirectLighting.tracePath || 'Baked / probe fallback'}</dd>
          <dt>Trace scale</dt>
          <dd>{diagnostics ? `${Math.round(diagnostics.indirectLighting.traceScale * 100)}%` : '0%'}</dd>
          <dt>Rays</dt>
          <dd>
            {diagnostics?.indirectLighting.giRays ?? 0} GI / {diagnostics?.indirectLighting.reflectionRays ?? 0}{' '}
            reflection
          </dd>
          <dt>Surface cache</dt>
          <dd>
            {diagnostics?.indirectLighting.residentSurfacePages ?? 0} /{' '}
            {diagnostics?.indirectLighting.surfaceCards ?? 0} pages/cards
          </dd>
          <dt>Distance fields</dt>
          <dd>{diagnostics?.indirectLighting.residentDistanceFieldPages ?? 0} resident pages</dd>
          <dt>Probe updates</dt>
          <dd>{diagnostics?.indirectLighting.radianceProbeUpdates ?? 0} this frame</dd>
          <dt>Hit rates</dt>
          <dd>
            {Math.round((diagnostics?.indirectLighting.screenHitRate ?? 0) * 100)}% screen /{' '}
            {Math.round((diagnostics?.indirectLighting.softwareHitRate ?? 0) * 100)}% software /{' '}
            {Math.round((diagnostics?.indirectLighting.hardwareHitRate ?? 0) * 100)}% hardware
          </dd>
        </dl>
      </section>
      {(diagnostics?.environment.fallback ||
        diagnostics?.shadows.fallback ||
        diagnostics?.indirectLighting.fallback) && (
        <div className="tool-warning">
          {diagnostics.environment.fallback || diagnostics.shadows.fallback || diagnostics.indirectLighting.fallback}
        </div>
      )}
    </section>
  );
}
