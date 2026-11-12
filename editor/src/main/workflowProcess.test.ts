import { describe, expect, it } from 'vitest';
import { runWorkflowProcess } from './workflowProcess';

describe('workflow subprocess', () => {
  const run = (script: string, signal = new AbortController().signal, timeoutMs = 5000) =>
    runWorkflowProcess({
      executable: process.execPath,
      args: ['-e', script],
      cwd: process.cwd(),
      signal,
      timeoutMs,
    });
  it('captures output without shell interpretation', async () => {
    expect(await run('process.stdout.write("hello; $world")')).toBe('hello; $world');
  });
  it('reports a nonzero exit and stderr', async () => {
    await expect(run('process.stderr.write("cook failed"); process.exit(2)')).rejects.toThrow('cook failed');
  });
  it('rejects cancellation before spawning', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(run('', controller.signal)).rejects.toThrow();
  });
  it('bounds hung commands', async () => {
    await expect(run('setInterval(() => {}, 1000)', undefined, 150)).rejects.toThrow('timed out');
  });
  it('bounds command output', async () => {
    await expect(run('process.stdout.write("x".repeat(2 * 1024 * 1024))')).rejects.toThrow('output exceeded');
  });
  it('cancels a running owned process', async () => {
    const controller = new AbortController();
    const pending = runWorkflowProcess({
      executable: process.execPath,
      args: ['-e', 'process.stdout.write("ready"); setInterval(() => {}, 1000)'],
      cwd: process.cwd(),
      signal: controller.signal,
      timeoutMs: 5000,
      onOutput: () => controller.abort(),
    });
    await expect(pending).rejects.toThrow('cancelled');
  });
});
