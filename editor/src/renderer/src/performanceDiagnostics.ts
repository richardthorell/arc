const enabled = import.meta.env.DEV || new URLSearchParams(window.location.search).has('perf');
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
        if (entry.duration >= 50) log(`long task ${entry.duration.toFixed(1)}ms at +${entry.startTime.toFixed(1)}ms`);
      }
    });
    observer.observe({ entryTypes: ['longtask'] });
  } catch {
    // Long-task observation is diagnostic-only and not supported by every runtime.
  }
};
