import type { BuildMonitorEvent, BuildMonitorRun } from './buildMonitorTypes';

/** Complete process output, retained separately from the bounded live diagnostic buffer. */
export type BuildRawLogChunk = {
  timestamp: string;
  stream: 'stdout' | 'stderr';
  text: string;
};

export type BuildHistoryPage = {
  projectGuid: string | null;
  total: number;
  offset: number;
  runs: BuildMonitorRun[];
  storageError?: string;
};

export type BuildHistoryChunk<T> = {
  items: T[];
  /** Byte offset in a JSONL file. Pass unchanged to read the next page. */
  nextCursor: number;
  hasMore: boolean;
};

export type BuildHistoryDetail = {
  run: BuildMonitorRun;
  logs: BuildHistoryChunk<BuildRawLogChunk>;
  events: BuildHistoryChunk<BuildMonitorEvent>;
};
