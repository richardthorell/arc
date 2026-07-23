import { useMemo, useState } from 'react';

import './profiler.css';

export type ProfilerJobSample = {
  sequence: number;
  name: string;
  priority: string;
  affinity: string;
  status: string;
  threadId: number;
  queuedNanoseconds: number;
  startedNanoseconds: number;
  completedNanoseconds: number;
};

export type ProfilerMemoryDomain = {
  domain: string;
  bytes: number;
  peakBytes: number;
  softLimit: number;
  hardLimit: number;
  pressure: boolean;
};

export type ProfilerAllocationGroup = {
  domain: string;
  tag: string;
  worldId: number;
  threadId: number;
  stackId: number;
  allocationCount: number;
  bytes: number;
};

export type ProfilerSnapshot = {
  timestampNanoseconds: number;
  memory: {
    bytes: number;
    softLimit: number;
    hardLimit: number;
    pressureEvents: number;
    domains: ProfilerMemoryDomain[];
    groups: ProfilerAllocationGroup[];
  };
  scheduler: {
    submitted: number;
    completed: number;
    stolen: number;
    cancelled: number;
    failed: number;
    queued: number;
    droppedEvents: number;
    jobs: ProfilerJobSample[];
  };
};

const formatBytes = (bytes: number) => {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
};

const durationMicroseconds = (job: ProfilerJobSample) =>
  Math.max(0, job.completedNanoseconds - job.startedNanoseconds) / 1000;

const unique = (values: Array<string | number>) => [...new Set(values)].sort();

export function ProfilerPanel({ samples }: { samples: ProfilerSnapshot[] }) {
  const [domain, setDomain] = useState('all');
  const [tag, setTag] = useState('all');
  const [world, setWorld] = useState('all');
  const [thread, setThread] = useState('all');
  const [stack, setStack] = useState('all');
  const latest = samples.at(-1);

  const chartPoints = useMemo(() => {
    if (samples.length < 2) return '';
    const maximum = Math.max(...samples.map((sample) => sample.memory.hardLimit || sample.memory.bytes), 1);
    return samples.map((sample, index) => {
      const x = (index / (samples.length - 1)) * 100;
      const y = 38 - (sample.memory.bytes / maximum) * 36;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(' ');
  }, [samples]);

  if (!latest) {
    return <div className="profiler-empty">
      <strong>Waiting for engine telemetry</strong>
      <span>Task timing and memory history will appear when the native host is connected.</span>
    </div>;
  }

  const groups = latest.memory.groups.filter((group) =>
    (domain === 'all' || group.domain === domain) &&
    (tag === 'all' || group.tag === tag) &&
    (world === 'all' || String(group.worldId) === world) &&
    (thread === 'all' || String(group.threadId) === thread) &&
    (stack === 'all' || String(group.stackId) === stack));
  const jobs = latest.scheduler.jobs.filter((job) => thread === 'all' || String(job.threadId) === thread);
  const budget = latest.memory.hardLimit || Math.max(latest.memory.bytes, 1);
  const memoryPercent = Math.min(100, latest.memory.bytes / budget * 100);

  return <section className="profiler-panel">
    <div className="profiler-summary">
      <article>
        <span>Memory</span>
        <strong>{formatBytes(latest.memory.bytes)}</strong>
        <small>{memoryPercent.toFixed(1)}% of {formatBytes(budget)}</small>
      </article>
      <article>
        <span>Queued Tasks</span>
        <strong>{latest.scheduler.queued}</strong>
        <small>{latest.scheduler.completed.toLocaleString()} completed</small>
      </article>
      <article>
        <span>Work Steals</span>
        <strong>{latest.scheduler.stolen.toLocaleString()}</strong>
        <small>{latest.scheduler.cancelled} cancelled · {latest.scheduler.failed} failed</small>
      </article>
      <article className={latest.memory.pressureEvents ? 'profiler-pressure' : ''}>
        <span>Pressure Events</span>
        <strong>{latest.memory.pressureEvents}</strong>
        <small>{latest.scheduler.droppedEvents} profiler events dropped</small>
      </article>
    </div>

    <div className="profiler-main">
      <div className="profiler-memory-column">
        <header><strong>Memory Timeline</strong><span>5 minute rolling window</span></header>
        <svg className="profiler-chart" viewBox="0 0 100 40" preserveAspectRatio="none" aria-label="Memory usage timeline">
          <path d="M0 39 H100 M0 20 H100 M0 1 H100" />
          {chartPoints && <polyline points={chartPoints} />}
        </svg>
        <div className="profiler-domain-list">
          {latest.memory.domains.filter((item) => item.bytes > 0 || item.peakBytes > 0).map((item) => {
            const limit = item.hardLimit || Math.max(item.peakBytes, item.bytes, 1);
            return <button type="button" key={item.domain} className={item.pressure ? 'pressure' : ''}
              onClick={() => setDomain(item.domain)}>
              <span>{item.domain}</span><b>{formatBytes(item.bytes)}</b>
              <i style={{ width: `${Math.min(100, item.bytes / limit * 100)}%` }} />
            </button>;
          })}
        </div>
      </div>

      <div className="profiler-detail-column">
        <div className="profiler-filters">
          <select value={domain} onChange={(event) => setDomain(event.target.value)}>
            <option value="all">All domains</option>
            {unique(latest.memory.groups.map((group) => group.domain)).map((value) => <option key={value}>{value}</option>)}
          </select>
          <select value={tag} onChange={(event) => setTag(event.target.value)}>
            <option value="all">All tags</option>
            {unique(latest.memory.groups.map((group) => group.tag)).map((value) => <option key={value}>{value}</option>)}
          </select>
          <select value={world} onChange={(event) => setWorld(event.target.value)}>
            <option value="all">All worlds</option>
            {unique(latest.memory.groups.map((group) => group.worldId)).map((value) =>
              <option key={value} value={String(value)}>World {value || 'global'}</option>)}
          </select>
          <select value={thread} onChange={(event) => setThread(event.target.value)}>
            <option value="all">All threads</option>
            {unique([
              ...latest.memory.groups.map((group) => group.threadId),
              ...latest.scheduler.jobs.map((job) => job.threadId),
            ]).map((value) => <option key={value} value={String(value)}>Thread {value}</option>)}
          </select>
          <select value={stack} onChange={(event) => setStack(event.target.value)}>
            <option value="all">All stacks</option>
            {unique(latest.memory.groups.map((group) => group.stackId)).map((value) =>
              <option key={value} value={String(value)}>{value ? `0x${value.toString(16)}` : 'Unsampled'}</option>)}
          </select>
        </div>

        <div className="profiler-tables">
          <div>
            <header><strong>Allocation Groups</strong><span>{groups.length} visible</span></header>
            <div className="profiler-table" data-testid="profiler-allocation-groups">
              {groups.slice(0, 100).map((group, index) => <div key={`${group.domain}-${group.tag}-${group.stackId}-${index}`}>
                <span>{group.tag}<small>{group.domain} · {group.allocationCount} allocs</small></span>
                <b>{formatBytes(group.bytes)}</b>
              </div>)}
            </div>
          </div>
          <div>
            <header><strong>Recent Task Scopes</strong><span>{jobs.length} visible</span></header>
            <div className="profiler-table">
              {jobs.slice(-100).reverse().map((job) => <div key={job.sequence}>
                <span>{job.name}<small>{job.affinity} · {job.priority} · {job.status}</small></span>
                <b>{durationMicroseconds(job).toFixed(1)} µs</b>
              </div>)}
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>;
}
