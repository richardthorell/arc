import type { LivePlayMessage } from '../common/livePlayProtocol';

export type LivePlayBaselineStatus = Extract<LivePlayMessage, { type: 'scene.baseline.status' }>['payload'];

type Pending = {
  resolve: (status: LivePlayBaselineStatus) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

const defaultActivationTimeoutMs = 15_000;

/**
 * L1.3b2: editor-side completion gate for a verified C9 cooked-baseline offer.
 * The authenticated control server validates the correlated status identity;
 * this gate enforces at-most-one in-flight request and lifecycle/timeout
 * handling without treating control-channel attachment as scene readiness.
 */
export class LivePlayBaselineOutcomeGate {
  private pending: Pending | null = null;

  waitForStatus(sendOffer: () => boolean, timeoutMs = defaultActivationTimeoutMs): Promise<LivePlayBaselineStatus> {
    if (this.pending) return Promise.reject(new Error('A Live Play scene baseline is already awaiting activation'));
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 120_000)
      return Promise.reject(new Error('Invalid Live Play scene activation timeout'));

    return new Promise<LivePlayBaselineStatus>((resolve, reject) => {
      const pending: Pending = {
        resolve,
        reject,
        timer: setTimeout(() => this.fail(pending, 'Live Play scene activation timed out'), timeoutMs),
      };
      this.pending = pending;
      try {
        if (!sendOffer()) this.fail(pending, 'Live Play could not offer a verified cooked scene baseline');
      } catch {
        this.fail(pending, 'Live Play could not offer a verified cooked scene baseline');
      }
    });
  }

  /** Called only for a validated, offer-correlated scene.baseline.status frame. */
  receive(status: LivePlayBaselineStatus): void {
    const pending = this.pending;
    if (!pending) return;
    this.pending = null;
    clearTimeout(pending.timer);
    if (status.status === 'active') pending.resolve(status);
    else pending.reject(new Error(`Live Play scene baseline rejected: ${status.reason}`));
  }

  cancel(reason = 'Live Play scene activation cancelled'): void {
    if (this.pending) this.fail(this.pending, reason);
  }

  private fail(pending: Pending, reason: string): void {
    if (this.pending !== pending) return;
    this.pending = null;
    clearTimeout(pending.timer);
    pending.reject(new Error(reason));
  }
}
