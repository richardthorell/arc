from pathlib import Path


def replace(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise RuntimeError(f"missing pattern in {path}: {old[:100]!r}")
    file.write_text(text.replace(old, new, 1))


Path('editor/src/main/performanceDiagnostics.ts').write_text('''const nowNs = () => process.hrtime.bigint();
const msBetween = (start: bigint, end = nowNs()) => Number(end - start) / 1_000_000;

export type PerformanceDiagnosticsOptions = {
  enabled: boolean;
  slowOperationMs?: number;
  logger?: (message: string) => void;
};

export class PerformanceDiagnostics {
  private readonly processStartedAt = nowNs();
  private readonly enabled: boolean;
  private readonly slowOperationMs: number;
  private readonly logger: (message: string) => void;

  constructor(options: PerformanceDiagnosticsOptions) {
    this.enabled = options.enabled;
    this.slowOperationMs = options.slowOperationMs ?? 50;
    this.logger = options.logger ?? console.info;
  }

  mark(label: string, detail = ''): void {
    if (!this.enabled) return;
    const suffix = detail ? ` ${detail}` : '';
    this.logger(`[ARC PERF] +${msBetween(this.processStartedAt).toFixed(1)}ms ${label}${suffix}`);
  }

  begin(operation: string): () => number {
    if (!this.enabled) return () => 0;
    const startedAt = nowNs();
    return () => {
      const durationMs = msBetween(startedAt);
      if (durationMs >= this.slowOperationMs)
        this.logger(`[ARC PERF] slow ${operation} ${durationMs.toFixed(1)}ms`);
      return durationMs;
    };
  }

  async measure<T>(operation: string, task: () => Promise<T>): Promise<T> {
    const finish = this.begin(operation);
    try {
      return await task();
    } finally {
      finish();
    }
  }
}
''')

Path('editor/src/main/performanceDiagnostics.test.ts').write_text('''import { describe, expect, it, vi } from 'vitest';
import { PerformanceDiagnostics } from './performanceDiagnostics';

describe('PerformanceDiagnostics', () => {
  it('stays silent when disabled', () => {
    const logger = vi.fn();
    const diagnostics = new PerformanceDiagnostics({ enabled: false, logger });
    diagnostics.mark('startup');
    diagnostics.begin('query')();
    expect(logger).not.toHaveBeenCalled();
  });

  it('records startup marks when enabled', () => {
    const logger = vi.fn();
    const diagnostics = new PerformanceDiagnostics({ enabled: true, slowOperationMs: 0, logger });
    diagnostics.mark('app.ready');
    expect(logger).toHaveBeenCalledWith(expect.stringContaining('[ARC PERF]'));
    expect(logger).toHaveBeenCalledWith(expect.stringContaining('app.ready'));
  });
});
''')

Path('editor/src/renderer/src/performanceDiagnostics.ts').write_text('''const enabled = import.meta.env.DEV || new URLSearchParams(window.location.search).has('perf');
const startedAt = performance.now();

const log = (message: string) => {
  if (enabled) console.info(`[ARC PERF] ${message}`);
};

export const installRendererPerformanceDiagnostics = (): void => {
  if (!enabled) return;
  log('renderer bootstrap started');

  requestAnimationFrame(() => {
    requestAnimationFrame(() => log(`renderer first settled paint ${Math.round(performance.now() - startedAt)}ms`));
  });

  if (!('PerformanceObserver' in window)) return;
  try {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.duration >= 50)
          log(`long task ${entry.duration.toFixed(1)}ms at +${entry.startTime.toFixed(1)}ms`);
      }
    });
    observer.observe({ entryTypes: ['longtask'] });
  } catch {
    // Long-task observation is diagnostic-only and not supported by every runtime.
  }
};
''')

replace('editor/src/main/main.ts', "import type { ArcCloneProjectRequest, ArcCreateProjectRequest } from '../common/projectTypes';\n", "import type { ArcCloneProjectRequest, ArcCreateProjectRequest } from '../common/projectTypes';\nimport { PerformanceDiagnostics } from './performanceDiagnostics';\n")
replace('editor/src/main/main.ts', "const isDevelopment = !app.isPackaged;\n", "const isDevelopment = !app.isPackaged;\nconst performanceDiagnostics = new PerformanceDiagnostics({\n  enabled: isDevelopment || process.env.ARC_EDITOR_PERF === '1' || app.commandLine.hasSwitch('perf-diagnostics'),\n  slowOperationMs: Number(process.env.ARC_EDITOR_PERF_SLOW_MS ?? 50),\n});\nperformanceDiagnostics.mark('main process started');\n")
replace('editor/src/main/main.ts', "{ resolve: (value: HostResponse) => void; reject: (reason: Error) => void }\n", "{ resolve: (value: HostResponse) => void; reject: (reason: Error) => void; finishTiming: () => number }\n")
replace('editor/src/main/main.ts', "      this.pending.set(requestId, { resolve, reject });\n", "      this.pending.set(requestId, {\n        resolve,\n        reject,\n        finishTiming: performanceDiagnostics.begin(`host ${message.kind} ${message.type}`),\n      });\n")
replace('editor/src/main/main.ts', "    this.pending.delete(maybeResponse.requestId);\n    pending.resolve(maybeResponse as HostResponse);\n", "    this.pending.delete(maybeResponse.requestId);\n    pending.finishTiming();\n    pending.resolve(maybeResponse as HostResponse);\n")
replace('editor/src/main/main.ts', "const createMainWindow = (): void => {\n", "const createMainWindow = (): void => {\n  performanceDiagnostics.mark('window creation started');\n")
replace('editor/src/main/main.ts', "  mainWindow.webContents.setWindowOpenHandler", "  performanceDiagnostics.mark('BrowserWindow created');\n  mainWindow.webContents.once('did-finish-load', () => performanceDiagnostics.mark('renderer did-finish-load'));\n\n  mainWindow.webContents.setWindowOpenHandler")
replace('editor/src/main/main.ts', "void app.whenReady().then(async () => {\n  Menu.setApplicationMenu(null);\n", "void app.whenReady().then(async () => {\n  performanceDiagnostics.mark('app ready');\n  Menu.setApplicationMenu(null);\n")
replace('editor/src/main/main.ts', "  hostClient = new ArcHostClient();\n", "  hostClient = new ArcHostClient();\n  performanceDiagnostics.mark('native host spawn requested');\n")
replace('editor/src/main/main.ts', "  recoveryService = new RecoveryService(\n", "  performanceDiagnostics.mark('core services constructed');\n  recoveryService = new RecoveryService(\n")
replace('editor/src/main/main.ts', '''    const opened = suppliedProject
      ? await projectService.open(suppliedProject)
      : await projectService.openOrCreateQuickStartProject(
          process.env.ARC_EDITOR_QUICK_START_PROJECT ?? path.join(app.getPath('userData'), 'QuickStartProject'),
        );
''', '''    const opened = await performanceDiagnostics.measure('startup project open', () =>
      suppliedProject
        ? projectService!.open(suppliedProject)
        : projectService!.openOrCreateQuickStartProject(
            process.env.ARC_EDITOR_QUICK_START_PROJECT ?? path.join(app.getPath('userData'), 'QuickStartProject'),
          ),
    );
''')
replace('editor/src/main/main.ts', "      await aiGateway.start();\n", "      await performanceDiagnostics.measure('AI gateway startup', () => aiGateway!.start());\n")
replace('editor/src/renderer/src/main.tsx', "import { App } from './App';\n", "import { App } from './App';\nimport { installRendererPerformanceDiagnostics } from './performanceDiagnostics';\n\ninstallRendererPerformanceDiagnostics();\n")
