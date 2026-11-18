/** BM7 measured native compile tasks. Times share the native tool's own invocation clock. */
export type BuildCompileTask = {
  /** Relative build output path, not a source filename. */
  output: string;
  startMs: number;
  endMs: number;
  durationMs: number;
};

export type BuildCompileTrace = {
  source: 'ninja-log' | 'msbuild-file-log';
  /** Native tool event clock, not the editor command's absolute start time. */
  clock: 'ninja-invocation' | 'msbuild-cl-events';
  tasks: BuildCompileTask[];
  truncated: boolean;
};
