import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { Focus, GitBranch, Maximize2, RotateCcw } from 'lucide-react';

import type { BuildCompileTask } from '../../../common/buildCompileTypes';
import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import { GraphViewportLayer, GraphWireLayer } from '../graph';
import {
  fitGraphViewport,
  panGraphViewport,
  zoomGraphViewportAt,
  type GraphViewportState,
} from '../graph/graphViewport';
import {
  buildExecutionGraphModel,
  formatDuration,
  packCompileTaskLanes,
  tasksForSelection,
  type BuildExecutionNode,
} from './buildExecutionGraphModel';

const DEFAULT_VIEWPORT: GraphViewportState = { x: 24, y: 20, zoom: 0.9 };
const VIEWPORT_LIMITS = { minZoom: 0.2, maxZoom: 1.6 };

const nodeCenter = (node: BuildExecutionNode, side: 'left' | 'right') => ({
  x: node.x + (side === 'right' ? node.width : 0),
  y: node.y + node.height / 2,
});

const wirePath = (from: BuildExecutionNode, to: BuildExecutionNode): string => {
  const source = nodeCenter(from, 'right');
  const destination = nodeCenter(to, 'left');
  const span = Math.max(36, Math.abs(destination.x - source.x) * 0.45);
  return `M ${source.x} ${source.y} C ${source.x + span} ${source.y}, ${destination.x - span} ${destination.y}, ${destination.x} ${destination.y}`;
};

function taskName(task: BuildCompileTask): string {
  return task.sourceFile ?? task.sourceFiles?.join(', ') ?? task.output;
}

function CompileTaskLanes({
  run,
  selected,
  onFilterOutput,
}: {
  run: BuildMonitorRun;
  selected: BuildExecutionNode | undefined;
  onFilterOutput?: (value: string) => void;
}) {
  const [selectedTask, setSelectedTask] = useState<BuildCompileTask | null>(null);
  const tasks = useMemo(() => tasksForSelection(run, selected), [run, selected]);
  const packed = useMemo(() => packCompileTaskLanes(tasks), [tasks]);
  const laneCount = Math.max(1, ...packed.map((entry) => entry.lane + 1));
  const endMs = Math.max(1, ...tasks.map((task) => task.endMs));
  const trace = run.compileTrace;
  const selectedTargetName = selected?.kind === 'target' ? selected.label : null;
  const active =
    selected?.kind === 'stage' && selected.label.toLowerCase() === 'compile' ? (trace?.activeTasks ?? []) : [];

  useEffect(() => setSelectedTask(null), [selected?.id]);

  if (!trace) {
    return <p className="build-execution-empty">Compiler timing is unavailable for this build.</p>;
  }
  if (!tasks.length && selectedTargetName && trace.source === 'msbuild-file-log') {
    return (
      <p className="build-execution-empty">
        MSBuild compiler tasks are measured but cannot be attributed safely to {selectedTargetName}. Select Compile to
        inspect all recorded CL tasks.
      </p>
    );
  }
  if (!tasks.length && !active.length) {
    return (
      <p className="build-execution-empty">
        No measured compiler tasks are attributed to {selected?.label ?? 'this selection'}.
      </p>
    );
  }

  return (
    <div className="build-execution-timing" aria-label="Compiler task timing">
      <div className="build-execution-timing-heading">
        <div>
          <strong>{selectedTargetName ?? 'Compile'} task timing</strong>
          <span>
            {trace.source === 'ninja-log' ? 'Ninja invocation clock' : 'MSBuild CL event clock'} · {tasks.length}{' '}
            measured
          </span>
        </div>
        {trace.truncated && <small>Bounded trace · some tasks omitted</small>}
      </div>
      <div className="build-execution-lanes" style={{ height: `${Math.max(42, laneCount * 30 + 20)}px` }}>
        <div className="build-execution-lane-axis">
          <span>0</span>
          <span>{formatDuration(endMs)}</span>
        </div>
        {packed.map(({ task, lane, index }) => (
          <button
            type="button"
            key={`${task.output}:${index}`}
            className="build-execution-task"
            style={{
              left: `${(task.startMs / endMs) * 100}%`,
              top: `${lane * 30 + 19}px`,
              width: `${Math.max(0.45, (task.durationMs / endMs) * 100)}%`,
            }}
            title={`${taskName(task)} · ${formatDuration(task.durationMs)}`}
            aria-label={`Inspect compile task: ${taskName(task)}`}
            onClick={() => setSelectedTask((current) => (current === task ? null : task))}
          />
        ))}
      </div>
      {active.length > 0 && (
        <div className="build-execution-active" aria-label="Active compiler tasks">
          <span>
            {active.length} CL {active.length === 1 ? 'task is' : 'tasks are'} active; no end timestamp or duration has
            been reported.
          </span>
          {active.map((task) => (
            <div key={task.output} title={task.sourceFiles?.join(', ') ?? task.output}>
              <i />
              <code>{task.output}</code>
            </div>
          ))}
        </div>
      )}
      {selectedTask && (
        <div className="build-execution-task-detail">
          <div>
            <strong>{taskName(selectedTask)}</strong>
            <span>{formatDuration(selectedTask.durationMs)}</span>
          </div>
          <code>{selectedTask.output}</code>
          {selectedTask.targetName && (
            <dl>
              <dt>CMake target</dt>
              <dd>{selectedTask.targetName}</dd>
            </dl>
          )}
          {trace.source === 'ninja-log' && onFilterOutput && (
            <button type="button" onClick={() => onFilterOutput(selectedTask.output)}>
              Find output text in build logs
            </button>
          )}
        </div>
      )}
      <p className="build-execution-clock-note">
        Native task timing uses its own tool clock and is not aligned to the editor workflow timeline.
      </p>
    </div>
  );
}

export function BuildExecutionGraph({
  run,
  onFilterOutput,
}: {
  run: BuildMonitorRun;
  onFilterOutput?: (value: string) => void;
}) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);
  const [showTargets, setShowTargets] = useState(Boolean(run.targetGraph));
  const [viewport, setViewport] = useState<GraphViewportState>(DEFAULT_VIEWPORT);
  const model = useMemo(() => buildExecutionGraphModel(run, showTargets), [run, showTargets]);
  const [selectedId, setSelectedId] = useState<string>(() => {
    const compile = run.stages.find((stage) => stage.name.toLowerCase() === 'compile');
    return compile ? `stage:${compile.id}` : `stage:${run.stages[0]?.id ?? 'unavailable'}`;
  });
  const selected = model.nodes.find((node) => node.id === selectedId) ?? model.nodes[0];

  const fit = useCallback(() => {
    const rect = canvasRef.current?.getBoundingClientRect();
    setViewport(
      fitGraphViewport(model.bounds, { width: rect?.width || 960, height: rect?.height || 430 }, 36, VIEWPORT_LIMITS),
    );
  }, [model.bounds]);

  useEffect(() => {
    const frame = requestAnimationFrame(fit);
    return () => cancelAnimationFrame(frame);
  }, [fit]);

  const nodes = new Map(model.nodes.map((node) => [node.id, node]));
  const wires = model.edges.flatMap((edge) => {
    const from = nodes.get(edge.from);
    const to = nodes.get(edge.to);
    return from && to
      ? [
          {
            id: edge.id,
            path: wirePath(from, to),
            tooltip: edge.kind === 'dependency' ? 'Target dependency' : undefined,
          },
        ]
      : [];
  });

  const pointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || (event.target as HTMLElement).closest('button')) return;
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const pointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    setViewport((current) =>
      panGraphViewport(current, { x: event.clientX - drag.x, y: event.clientY - drag.y }, VIEWPORT_LIMITS),
    );
    dragRef.current = { ...drag, x: event.clientX, y: event.clientY };
  };
  const pointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
  };

  return (
    <section className="build-execution" aria-label="Build execution graph">
      <div className="build-execution-toolbar">
        <div>
          <GitBranch size={15} aria-hidden="true" />
          <strong>Execution graph</strong>
          <span>{run.targetGraph ? `${run.targetGraph.nodes.length} native targets` : 'Workflow stages'}</span>
          {run.targetGraph?.truncated && <small>Graph truncated</small>}
        </div>
        <div>
          {run.targetGraph && (
            <button type="button" onClick={() => setShowTargets((value) => !value)}>
              <Focus size={14} /> {showTargets ? 'Collapse targets' : 'Expand targets'}
            </button>
          )}
          <button type="button" onClick={fit} aria-label="Fit execution graph">
            <Maximize2 size={14} /> Fit
          </button>
          <button type="button" onClick={() => setViewport(DEFAULT_VIEWPORT)} aria-label="Reset graph view">
            <RotateCcw size={14} /> Reset
          </button>
        </div>
      </div>
      <div
        ref={canvasRef}
        className="build-execution-canvas"
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerUp={pointerUp}
        onPointerCancel={pointerUp}
        onWheel={(event) => {
          event.preventDefault();
          const rect = event.currentTarget.getBoundingClientRect();
          setViewport((current) =>
            zoomGraphViewportAt(
              current,
              current.zoom * (event.deltaY > 0 ? 0.9 : 1.1),
              { x: event.clientX - rect.left, y: event.clientY - rect.top },
              VIEWPORT_LIMITS,
            ),
          );
        }}
      >
        <GraphViewportLayer className="build-execution-transform" viewport={viewport} size={4096}>
          <GraphWireLayer className="build-execution-wires" wires={wires} size={4096} />
          {model.nodes.map((node) => (
            <button
              type="button"
              key={node.id}
              className={`build-execution-node is-${node.kind} is-${node.state}${selected?.id === node.id ? ' is-selected' : ''}${node.root ? ' is-root' : ''}`}
              style={{ left: node.x, top: node.y, width: node.width, height: node.height }}
              aria-pressed={selected?.id === node.id}
              onClick={() => setSelectedId(node.id)}
            >
              <span>{node.kind}</span>
              <strong>{node.label}</strong>
              <small>{node.detail}</small>
            </button>
          ))}
        </GraphViewportLayer>
      </div>
      <CompileTaskLanes run={run} selected={selected} onFilterOutput={onFilterOutput} />
    </section>
  );
}
