import { describe, expect, it, vi } from 'vitest';
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
