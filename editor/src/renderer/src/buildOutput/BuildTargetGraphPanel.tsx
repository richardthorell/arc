import { useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, GitBranch, PackageCheck } from 'lucide-react';

import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import type { BuildTargetGraph, BuildTargetNode } from '../../../common/buildTargetTypes';

type VisibleTarget = { node: BuildTargetNode; depth: number; shared: boolean };
const safeTime = (value: string): number | null => {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
};

function graphRows(graph: BuildTargetGraph, expanded: Set<string>): VisibleTarget[] {
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  const visited = new Set<string>();
  const rows: VisibleTarget[] = [];
  const visit = (id: string, depth: number): void => {
    if (rows.length >= 256) return;
    const node = byId.get(id);
    if (!node) return;
    const shared = visited.has(id);
    rows.push({ node, depth, shared });
    if (shared || !expanded.has(id)) return;
    visited.add(id);
    for (const dependency of node.dependencies) visit(dependency, depth + 1);
  };
  visit(graph.rootTargetId, 0);
  return rows;
}

const formatOffset = (milliseconds: number): string =>
  milliseconds < 1000 ? `${Math.round(milliseconds)} ms` : `${(milliseconds / 1000).toFixed(1)} s`;

export function BuildTargetGraphPanel({ run, nowMs }: { run: BuildMonitorRun; nowMs: number }) {
  const graph = run.targetGraph;
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(graph ? [graph.rootTargetId] : []));
  const rows = useMemo(() => (graph ? graphRows(graph, expanded) : []), [graph, expanded]);

  if (!graph) {
    return (
      <section className="build-monitor-targets" aria-label="Target dependency graph">
        <div className="build-monitor-section-title">
          <h4>Target dependencies</h4>
        </div>
        <p className="build-monitor-empty">
          {run.source === 'command'
            ? 'Target graph is not available for legacy CLI commands.'
            : run.state === 'running'
              ? 'Waiting for CMake configure to report target dependencies…'
              : 'CMake target metadata was not available for this build.'}
        </p>
      </section>
    );
  }

  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const observed = graph.nodes.filter((node) => node.artifactObservedAt).length;
  const start = safeTime(run.startedAt) ?? 0;
  const spanMs = Math.max(1, run.durationMs ?? Math.max(0, nowMs - start));
  const toggle = (id: string) => {
    setExpanded((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <section className="build-monitor-targets" aria-label="Target dependency graph">
      <div className="build-monitor-section-title">
        <h4>
          <GitBranch size={15} aria-hidden="true" /> CMake target dependencies
        </h4>
        <span>
          {observed} artifact output{observed === 1 ? '' : 's'} observed · {graph.nodes.length} targets
        </span>
      </div>
      <p className="build-monitor-target-note">
        Dependencies come from CMake File API ({graph.configuration}). Markers show explicitly reported native
        artifacts, not an estimate of completed targets. Up-to-date targets may emit no output.
      </p>
      <div className="build-monitor-target-axis" aria-hidden="true">
        <span>Dependency tree</span>
        <span>Build elapsed time →</span>
        <span>Observation</span>
      </div>
      <div className="build-monitor-target-list" role="tree" aria-label="CMake target dependencies">
        {rows.map(({ node, depth, shared }, index) => {
          const hasChildren = node.dependencies.some((id) => nodes.has(id)) && !shared;
          const marker = safeTime(node.artifactObservedAt ?? '');
          const startOffset = marker === null ? null : Math.max(0, marker - start);
          const percentage = startOffset === null ? null : Math.min(100, (startOffset / spanMs) * 100);
          const kind = node.satisfiedAt ? 'satisfied' : node.artifactObservedAt ? 'observed' : 'unobserved';
          return (
            <div
              key={`${node.id}-${index}`}
              className="build-monitor-target-row"
              role="treeitem"
              aria-level={Math.min(depth + 1, 32)}
              aria-expanded={hasChildren ? expanded.has(node.id) : undefined}
            >
              <div className="build-monitor-target-name" style={{ paddingLeft: `${Math.min(depth, 7) * 15}px` }}>
                {hasChildren ? (
                  <button
                    type="button"
                    aria-label={`${expanded.has(node.id) ? 'Collapse' : 'Expand'} ${node.name}`}
                    onClick={() => toggle(node.id)}
                  >
                    {expanded.has(node.id) ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  </button>
                ) : (
                  <span className="build-monitor-target-spacer" />
                )}
                <span title={`${node.name} · ${node.type}`}>{node.name}</span>
                {shared && <small title="Shared dependency already listed">shared</small>}
              </div>
              <div className="build-monitor-target-track">
                {percentage !== null && (
                  <span
                    className="build-monitor-target-marker"
                    style={{ left: `${percentage}%` }}
                    title={`Artifact output observed ${formatOffset(startOffset!)} into build`}
                    aria-label={`Artifact output observed ${formatOffset(startOffset!)} into build`}
                  />
                )}
              </div>
              <span className={`build-monitor-target-result is-${kind}`}>
                {kind === 'satisfied' ? <PackageCheck size={12} /> : null}
                {kind === 'satisfied' ? 'Satisfied' : kind === 'observed' ? 'Output seen' : 'Not reported'}
              </span>
            </div>
          );
        })}
      </div>
      {graph.truncated && (
        <p className="build-monitor-target-note">
          Graph limited to 128 reachable targets. Some dependencies are omitted.
        </p>
      )}
    </section>
  );
}
