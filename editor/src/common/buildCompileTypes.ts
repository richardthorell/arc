/** BM7 measured native compile tasks. Times share the native tool's own invocation clock. */
export type BuildCompileTask = {
  /** Relative build output path, not a source filename. */
  output: string;
  startMs: number;
  endMs: number;
  durationMs: number;
  /** Project-relative path from an unambiguous CMake compilation-database entry. */
  sourceFile?: string;
  /** CMake codemodel-verified target, when part of the tracked graph. */
  targetName?: string;
};

export type BuildCompileTrace = {
  source: 'ninja-log' | 'msbuild-file-log';
  /** Native tool event clock, not the editor command's absolute start time. */
  clock: 'ninja-invocation' | 'msbuild-cl-events';
  tasks: BuildCompileTask[];
  truncated: boolean;
};
