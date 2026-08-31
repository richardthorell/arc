const nowNs = () => process.hrtime.bigint();
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
      if (durationMs >= this.slowOperationMs) this.logger(`[ARC PERF] slow ${operation} ${durationMs.toFixed(1)}ms`);
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
