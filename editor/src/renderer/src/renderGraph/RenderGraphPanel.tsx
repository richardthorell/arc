import { useEffect, useMemo, useState } from 'react';
import { GitCompareArrows, Pause, Play, RefreshCw } from 'lucide-react';

import type { HostResponse } from '../inspector/inspectorTypes';
import { UiButton, UiIconButton } from '../ui';
import type { EditorDiagnosticsSnapshot } from '../tools/toolTypes';

import '../tools/tools.css';

const bytes = (value: number) =>
  value >= 1024 * 1024 ? `${(value / (1024 * 1024)).toFixed(1)} MiB` : `${(value / 1024).toFixed(1)} KiB`;

const graphNodeWidth = 154;
const graphNodeGap = 54;
const graphLeft = 30;
const graphTop = 34;
const resourceTop = 132;
const resourceLaneHeight = 30;

export function RenderGraphPanel({
  fixtureSnapshot,
  queryHost = true,
}: {
  fixtureSnapshot?: EditorDiagnosticsSnapshot;
  queryHost?: boolean;
} = {}) {
  const [snapshot, setSnapshot] = useState<EditorDiagnosticsSnapshot | null>(fixtureSnapshot ?? null);
  const [pinned, setPinned] = useState<EditorDiagnosticsSnapshot | null>(null);
  const [live, setLive] = useState(queryHost);
  const [filter, setFilter] = useState('');
  const [error, setError] = useState('');

  const refresh = async () => {
    if (!queryHost) return;
    const response = (await window.arc.host.query('gateway.diagnostics')) as HostResponse<EditorDiagnosticsSnapshot>;
    if (!response.succeeded || !response.payload) {
      setError(response.error || 'Render diagnostics are unavailable');
      return;
    }
    setSnapshot(response.payload);
    setError('');
  };

  useEffect(() => {
    if (!queryHost) return;
    void refresh();
    if (!live) return;
    const timer = window.setInterval(() => void refresh(), 500);
    return () => window.clearInterval(timer);
    // refresh is intentionally scoped to the current host/query mode.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live, queryHost]);

  const resources = useMemo(() => {
    const query = filter.trim().toLocaleLowerCase();
    return (snapshot?.graph.resources ?? []).filter(
      (resource) =>
        !query ||
        resource.name.toLocaleLowerCase().includes(query) ||
        resource.format.toLocaleLowerCase().includes(query),
    );
  }, [filter, snapshot]);
  const visiblePasses = useMemo(() => {
    const query = filter.trim().toLocaleLowerCase();
    const matchingResourcePasses = new Set<number>();
    if (query)
      for (const resource of resources)
        for (let index = resource.firstPass; index <= resource.lastPass; ++index) matchingResourcePasses.add(index);
    return (snapshot?.graph.executedPasses ?? []).filter(
      (pass, index) => !query || pass.toLocaleLowerCase().includes(query) || matchingResourcePasses.has(index),
    );
  }, [filter, resources, snapshot]);
  const timing = new Map(snapshot?.passes.map((pass) => [pass.name, pass.milliseconds]) ?? []);
  const graphWidth = Math.max(
    520,
    graphLeft * 2 + (snapshot?.graph.executedPasses.length ?? 0) * (graphNodeWidth + graphNodeGap) - graphNodeGap,
  );
  const graphHeight = Math.max(260, resourceTop + resources.length * resourceLaneHeight + 40);

  return (
    <section className="production-tool-panel render-graph-panel">
      <header className="tool-panel-toolbar">
        <strong>Executed Render Graph</strong>
        <input
          aria-label="Filter render graph"
          placeholder="Filter passes or resources…"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        />
        <UiIconButton
          disabled={!queryHost}
          label={live ? 'Pause graph updates' : 'Resume graph updates'}
          onClick={() => setLive(!live)}
        >
          {live ? <Pause size={14} /> : <Play size={14} />}
        </UiIconButton>
        <UiIconButton disabled={!queryHost} label="Refresh graph" onClick={() => void refresh()}>
          <RefreshCw size={14} />
        </UiIconButton>
        <UiButton onClick={() => setPinned(snapshot)} variant="toolbar">
          Pin frame
        </UiButton>
      </header>
      {snapshot && (
        <div className="tool-summary-strip">
          <span>Frame {snapshot.frameIndex}</span>
          <span>{snapshot.renderer.path}</span>
          <span>{snapshot.renderer.qualityTier}</span>
          <span>{Math.round(snapshot.renderer.renderScale * 100)}%</span>
          <span>{snapshot.graph.resourceCount} resources</span>
          <span>{snapshot.graph.physicalResourceCount ?? snapshot.graph.resourceCount} physical</span>
          <span>{snapshot.graph.aliasCount ?? 0} aliases</span>
          <span>{snapshot.graph.culledPasses?.length ?? 0} culled</span>
          <span>{snapshot.graph.barrierCount} barriers</span>
          <span>{bytes(snapshot.graph.estimatedTransientBytes)} transient</span>
        </div>
      )}
      {!!snapshot?.graph.submissions?.length && (
        <div className="tool-summary-strip" aria-label="Render graph queue submissions">
          {snapshot.graph.submissions.map((submission, index) => (
            <span key={`${submission.queue}-${index}`}>
              {submission.queue}: {submission.passCount} passes · {submission.waitCount} waits · signal{' '}
              {submission.signalValue}
            </span>
          ))}
          {!!snapshot.graph.histories?.length && (
            <span>
              {snapshot.graph.histories.length} histories
              {snapshot.graph.histories.some((history) => history.invalidated) ? ' · reset' : ''}
            </span>
          )}
        </div>
      )}
      {snapshot?.textureStreaming && (
        <div className="tool-summary-strip" aria-label="Texture streaming diagnostics">
          <span>
            Texture GPU {bytes(snapshot.textureStreaming.gpuResidentBytes)} /{' '}
            {bytes(snapshot.textureStreaming.gpuBudgetBytes)}
          </span>
          <span>
            CPU {bytes(snapshot.textureStreaming.cpuCachedBytes)} / {bytes(snapshot.textureStreaming.cpuBudgetBytes)}
          </span>
          <span>
            {snapshot.textureStreaming.residentMips} mips · {snapshot.textureStreaming.residentPages} pages resident
          </span>
          <span>
            {snapshot.textureStreaming.requestedMips + snapshot.textureStreaming.requestedPages} requested ·{' '}
            {snapshot.textureStreaming.failedMips + snapshot.textureStreaming.failedPages} failed
          </span>
          <span>Hit {(snapshot.textureStreaming.cacheHitRate * 100).toFixed(1)}%</span>
          <span>{snapshot.textureStreaming.evictions} evictions</span>
          {snapshot.textureStreaming.feedbackOverflow > 0 && (
            <span>{snapshot.textureStreaming.feedbackOverflow} feedback overflow</span>
          )}
        </div>
      )}
      {snapshot?.gpuScene && (
        <div className="tool-summary-strip" aria-label="GPU Scene resource tables">
          <span>
            Tables {snapshot.gpuScene.geometryTableEntries} geometry · {snapshot.gpuScene.materialTableEntries}{' '}
            materials · {snapshot.gpuScene.textureTableEntries} textures
          </span>
          <span>
            Heaps {bytes(snapshot.gpuScene.sharedVertexHeapBytes)} vertices ·{' '}
            {bytes(snapshot.gpuScene.sharedIndexHeapBytes)} indices
          </span>
          <span>
            Sparse upload {bytes(snapshot.gpuScene.uploadedBytes)} in {snapshot.gpuScene.uploadedRanges} ranges
          </span>
          <span>
            Visibility {snapshot.gpuScene.visibleInstances}/{snapshot.gpuScene.candidateInstances} ·{' '}
            {snapshot.gpuScene.activePipelineBins} bins · {snapshot.gpuScene.indirectCommands} commands
          </span>
          {snapshot.gpuScene.overflowRecords > 0 && (
            <span>{snapshot.gpuScene.overflowRecords} overflow fallback records</span>
          )}
        </div>
      )}
      {pinned && snapshot && (
        <div className="tool-comparison">
          <GitCompareArrows size={14} />
          Comparing frame {pinned.frameIndex} to {snapshot.frameIndex}: passes{' '}
          {snapshot.graph.executedPasses.length - pinned.graph.executedPasses.length >= 0 ? '+' : ''}
          {snapshot.graph.executedPasses.length - pinned.graph.executedPasses.length}, memory{' '}
          {bytes(snapshot.graph.estimatedTransientBytes - pinned.graph.estimatedTransientBytes)}
          <button onClick={() => setPinned(null)} type="button">
            Clear
          </button>
        </div>
      )}
      <div className="render-graph-workspace">
        <div className="render-graph-canvas">
          {snapshot && visiblePasses.length > 0 ? (
            <svg
              aria-label={`Executed graph for frame ${snapshot.frameIndex}`}
              height={graphHeight}
              role="img"
              viewBox={`0 0 ${graphWidth} ${graphHeight}`}
              width={graphWidth}
            >
              <defs>
                <marker id="render-graph-arrow" markerHeight="7" markerWidth="7" orient="auto" refX="6" refY="3.5">
                  <path d="M0,0 L7,3.5 L0,7 Z" fill="#5285a2" />
                </marker>
              </defs>
              {(snapshot.graph.transitions ?? []).map((transition, index) => {
                const fromX = graphLeft + transition.beforePass * (graphNodeWidth + graphNodeGap) + graphNodeWidth / 2;
                const toX = graphLeft + transition.afterPass * (graphNodeWidth + graphNodeGap) + graphNodeWidth / 2;
                return (
                  <path
                    className="render-graph-transition"
                    d={`M ${fromX} ${graphTop + 58} C ${fromX} 112, ${toX} 112, ${toX} ${graphTop + 58}`}
                    key={`${transition.resource}-${transition.beforePass}-${transition.afterPass}-${index}`}
                    markerEnd="url(#render-graph-arrow)"
                  >
                    <title>
                      {transition.resource}: {transition.before} → {transition.after} ({transition.beforeQueue} →{' '}
                      {transition.afterQueue})
                    </title>
                  </path>
                );
              })}
              {snapshot.graph.executedPasses.map((pass, index) => {
                const x = graphLeft + index * (graphNodeWidth + graphNodeGap);
                const visible = visiblePasses.includes(pass);
                return (
                  <g className="render-graph-svg-node" data-filtered={!visible} key={`${pass}-${index}`}>
                    <rect height="58" rx="4" width={graphNodeWidth} x={x} y={graphTop} />
                    <circle cx={x + 18} cy={graphTop + 18} r="10" />
                    <text className="render-graph-node-index" x={x + 18} y={graphTop + 22}>
                      {index + 1}
                    </text>
                    <text className="render-graph-node-name" x={x + 34} y={graphTop + 21}>
                      {pass.length > 18 ? `${pass.slice(0, 17)}…` : pass}
                      <title>{pass}</title>
                    </text>
                    <text className="render-graph-node-timing" x={x + 12} y={graphTop + 44}>
                      {timing.has(pass) ? `${timing.get(pass)?.toFixed(3)} ms` : 'GPU timing pending'}
                    </text>
                  </g>
                );
              })}
              {resources.map((resource, index) => {
                const startX = graphLeft + resource.firstPass * (graphNodeWidth + graphNodeGap) + graphNodeWidth / 2;
                const endX = graphLeft + resource.lastPass * (graphNodeWidth + graphNodeGap) + graphNodeWidth / 2;
                const y = resourceTop + index * resourceLaneHeight;
                return (
                  <g className="render-graph-resource-lifetime" key={`${resource.name}-${resource.physicalResource}`}>
                    <text x={graphLeft} y={y - 6}>
                      {resource.name} · P{resource.physicalResource} · {bytes(resource.estimatedBytes)}
                      {resource.aliased ? ' · alias' : ''}
                    </text>
                    <line x1={startX} x2={Math.max(startX + 2, endX)} y1={y + 5} y2={y + 5} />
                    <circle cx={startX} cy={y + 5} r="3" />
                    <circle cx={Math.max(startX + 2, endX)} cy={y + 5} r="3" />
                  </g>
                );
              })}
            </svg>
          ) : (
            <div className="tool-empty">No executed passes match this filter.</div>
          )}
        </div>
        <aside className="render-resource-list">
          <h3>Physical resources</h3>
          {resources.map((resource) => (
            <article key={`${resource.name}-${resource.firstPass}`}>
              <strong>{resource.name}</strong>
              <span>
                P{resource.physicalResource} · pass {resource.firstPass}–{resource.lastPass}
                {resource.aliased ? ' · alias' : ''}
              </span>
              <small>
                {resource.format} · {bytes(resource.estimatedBytes)}
              </small>
            </article>
          ))}
          {!resources.length && <p>Detailed resource lifetimes will appear after the native backend publishes them.</p>}
        </aside>
      </div>
      {snapshot?.renderer.fallbackReasons.map((reason) => (
        <div className="tool-warning" key={reason}>
          {reason}
        </div>
      ))}
      {snapshot?.textureStreaming?.fallback && <div className="tool-warning">{snapshot.textureStreaming.fallback}</div>}
      {error && <div className="tool-error">{error}</div>}
    </section>
  );
}
