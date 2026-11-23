import {
  applyLivePlayHierarchy,
  emptyLivePlayHierarchy,
  type HierarchyChangeMessage,
  type HierarchySnapshotMessage,
  type LivePlayHierarchyState,
} from '../common/livePlayHierarchyState';
import type { WindowsLivePlayState } from '../common/livePlaySessionTypes';

export type HierarchyRecoveryReason = 'reconnected' | 'gap' | 'overflow';
export type HierarchyRecoveryStatus = {
  stale: boolean;
  reason: HierarchyRecoveryReason | null;
  attempts: number;
};

const pageTimeoutMs = 1600;
const retryIntervalMs = 2400;
const maxRetries = 4;

/** Owns one active native session's hierarchy, not the Authoring World. */
export class LivePlayHierarchyRecovery {
  private hierarchyState = emptyLivePlayHierarchy();
  private connection: WindowsLivePlayState = 'stopped';
  private lastPageAtMs: number | null = null;
  private lastRequestAtMs: number | null = null;
  private attempts = 0;
  private reason: HierarchyRecoveryReason | null = null;

  constructor(
    private readonly request: (reason: HierarchyRecoveryReason) => boolean,
    private readonly now: () => number = Date.now,
  ) {}

  snapshot(): LivePlayHierarchyState {
    return this.hierarchyState;
  }

  status(): HierarchyRecoveryStatus {
    return {
      stale: this.hierarchyState.requiresSnapshot,
      reason: this.reason,
      attempts: this.attempts,
    };
  }

  reset(): void {
    this.hierarchyState = emptyLivePlayHierarchy();
    this.lastPageAtMs = null;
    this.lastRequestAtMs = null;
    this.attempts = 0;
    this.reason = null;
  }

  connectionState(next: WindowsLivePlayState): void {
    const wasDisconnected = this.connection === 'disconnected' || this.connection === 'failed';
    this.connection = next;
    if (next === 'stopped' || next === 'starting' || next === 'failed') {
      this.reset();
      return;
    }
    if (next === 'disconnected') {
      this.hierarchyState = { ...this.hierarchyState, pending: null, requiresSnapshot: true };
      this.lastPageAtMs = null;
      this.lastRequestAtMs = null;
      this.attempts = 0;
      this.reason = 'reconnected';
    } else if (next === 'connected' && wasDisconnected) {
      // The old world revision cannot serve as a base for a resumed socket.
      this.hierarchyState = { ...this.hierarchyState, pending: null, requiresSnapshot: true };
      this.lastPageAtMs = null;
      this.lastRequestAtMs = null;
      this.attempts = 0;
      this.ask('reconnected');
    }
  }

  observe(message: HierarchySnapshotMessage | HierarchyChangeMessage): void {
    if (this.connection !== 'connected') return;
    const before = this.hierarchyState;
    const next = applyLivePlayHierarchy(before, message);
    this.hierarchyState = next;

    if (next.pending !== before.pending && next.pending) this.lastPageAtMs = this.now();
    if (next.pending === null) this.lastPageAtMs = null;

    if (next.requiresSnapshot) {
      if (!before.requiresSnapshot) this.ask('gap');
    } else if (next.hierarchy && next.hierarchy !== before.hierarchy && !next.pending) {
      this.lastRequestAtMs = null;
      this.attempts = 0;
      this.reason = null;
    }
  }

  /** Call on a bounded timer in editor main (no renderer or native ECS reads). */
  tick(): void {
    if (this.connection !== 'connected') return;
    if (this.hierarchyState.pending && this.lastPageAtMs !== null && this.now() - this.lastPageAtMs >= pageTimeoutMs) {
      this.hierarchyState = { ...this.hierarchyState, pending: null, requiresSnapshot: true };
      this.lastPageAtMs = null;
      this.ask('overflow');
    }
    if (
      this.hierarchyState.requiresSnapshot &&
      this.attempts < maxRetries &&
      (this.lastRequestAtMs === null || this.now() - this.lastRequestAtMs >= retryIntervalMs)
    )
      this.ask(this.reason ?? 'gap');
  }

  private ask(reason: HierarchyRecoveryReason): void {
    if (this.connection !== 'connected' || this.attempts >= maxRetries) return;
    if (this.lastRequestAtMs !== null && this.now() - this.lastRequestAtMs < retryIntervalMs) return;
    if (!this.request(reason)) return;
    this.attempts++;
    this.lastRequestAtMs = this.now();
    this.reason = reason;
  }
}
