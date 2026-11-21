import { describe, expect, it, vi } from 'vitest';
import { LivePlayBaselineOutcomeGate, type LivePlayBaselineStatus } from './livePlayBaselineOutcomeGate';

const acknowledged: LivePlayBaselineStatus = {
  requestId: 'offer-1',
  sceneGuid: '22222222-2222-2222-2222-222222222222',
  cookedGeneration: 2,
  cookedReleaseSha256: 'a'.repeat(64),
  status: 'active',
  reason: 'none',
};

describe('LivePlayBaselineOutcomeGate', () => {
  it('awaits the correlated native activation result and permits the next offer', async () => {
    const gate = new LivePlayBaselineOutcomeGate();
    const send = vi.fn(() => true);
    const pending = gate.waitForStatus(send);
    expect(send).toHaveBeenCalledOnce();
    await expect(gate.waitForStatus(send)).rejects.toThrow(/already awaiting/);
    gate.receive(acknowledged);
    await expect(pending).resolves.toMatchObject({ status: 'active', cookedGeneration: 2 });
    expect(send).toHaveBeenCalledOnce();
    const rejected = gate.waitForStatus(send);
    gate.receive({ ...acknowledged, status: 'rejected', reason: 'incompatible' });
    await expect(rejected).rejects.toThrow(/rejected: incompatible/);
    expect(send).toHaveBeenCalledTimes(2);
  });

  it('does not report readiness when offering fails or the native runtime disconnects', async () => {
    const gate = new LivePlayBaselineOutcomeGate();
    await expect(gate.waitForStatus(() => false)).rejects.toThrow(/could not offer/);
    await expect(
      gate.waitForStatus(() => {
        throw new Error('sensitive internal error');
      }),
    ).rejects.toThrow(/could not offer/);
    const pending = gate.waitForStatus(() => true);
    gate.cancel('Live Play stopped before scene activation');
    await expect(pending).rejects.toThrow(/stopped/);
    gate.receive(acknowledged);
    const later = gate.waitForStatus(() => true);
    gate.receive(acknowledged);
    await expect(later).resolves.toMatchObject({ status: 'active' });
  });

  it('times out in a bounded interval and does not let stale messages complete another request', async () => {
    vi.useFakeTimers();
    try {
      const gate = new LivePlayBaselineOutcomeGate();
      const expired = gate.waitForStatus(() => true, 50);
      const assertion = expect(expired).rejects.toThrow(/timed out/);
      await vi.advanceTimersByTimeAsync(50);
      await assertion;
      const current = gate.waitForStatus(() => true, 50);
      gate.cancel();
      await expect(current).rejects.toThrow(/cancelled/);
      await expect(gate.waitForStatus(() => true, 0)).rejects.toThrow(/Invalid/);
    } finally {
      vi.useRealTimers();
    }
  });
});
