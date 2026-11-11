export type ArcBuildState = 'idle' | 'configuring' | 'building' | 'cleaning' | 'failed' | 'succeeded' | 'cancelled';

export type ArcBuildDiagnostic = {
  sequence: number;
  severity: 'info' | 'warning' | 'error';
  message: string;
  file?: string;
  line?: number;
  column?: number;
  category?: 'compiler' | 'codegen' | 'linker' | 'module';
};

/** Completed CLI/CMake commands in the current editor session (not estimated compile progress). */
export type ArcBuildRun = {
  id: number;
  kind: 'build' | 'configure' | 'clean' | 'ide';
  configuration: string;
  startedAt: string;
  durationMs: number;
  result: 'succeeded' | 'failed' | 'cancelled';
  warningCount: number;
  errorCount: number;
};

export type ArcBuildSnapshot = {
  revision: number;
  state: ArcBuildState;
  configuration: string;
  buildRequired: boolean;
  reloadRequired: boolean;
  restartRequired: boolean;
  diagnostics: ArcBuildDiagnostic[];
  history?: ArcBuildRun[];
  startedAt?: string;
  completedAt?: string;
  command?: string;
};

export type ArcBuildRequest = {
  action: 'configure' | 'build' | 'rebuild' | 'clean' | 'cancel' | 'reload' | 'openIde';
  configuration?: string;
  ide?: 'visual-studio' | 'vscode' | 'clion';
};
