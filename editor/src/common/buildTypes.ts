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

export type ArcBuildSnapshot = {
  revision: number;
  state: ArcBuildState;
  configuration: string;
  buildRequired: boolean;
  reloadRequired: boolean;
  restartRequired: boolean;
  diagnostics: ArcBuildDiagnostic[];
  startedAt?: string;
  completedAt?: string;
  command?: string;
};

export type ArcBuildRequest = {
  action: 'configure' | 'build' | 'rebuild' | 'clean' | 'cancel' | 'reload' | 'openIde';
  configuration?: string;
  ide?: 'visual-studio' | 'vscode' | 'clion';
};
