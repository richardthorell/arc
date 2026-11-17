/** BM7 measured native compile tasks. Times share the native tool's own invocation clock. */
export type BuildCompileTask = {
  /** Relative build output path, not a source filename. */
  output: string;
  startMs: number;
  endMs: number;
  durationMs: number;
};

export type BuildCompileTrace = {
  source: 'ninja-log';
  /** Ninja's offsets do not include CMake configuration or CLI startup time. */
  clock: 'ninja-invocation';
  tasks: BuildCompileTask[];
  truncated: boolean;
};
