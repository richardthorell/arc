import {
  BUILT_IN_AGENT_CLIENT_ID,
  BUILT_IN_AGENT_CLIENT_NAME,
  type BuiltInAgentCapabilities,
  type BuiltInAgentEvent,
} from '../common/builtInAgentTypes';
import { EditorAgentHarness } from './editorAgentHarness';

const stringArray = (value: unknown): value is readonly string[] =>
  Array.isArray(value) && value.every((entry) => typeof entry === 'string' && entry.length > 0);

const capabilitiesFromHarness = (value: unknown): BuiltInAgentCapabilities => {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('Editor agent harness returned an invalid capability snapshot');

  const snapshot = value as Record<string, unknown>;
  if (!stringArray(snapshot.operations) || !stringArray(snapshot.editActions))
    throw new Error('Editor agent harness capability snapshot is missing operations or edit actions');

  return snapshot as BuiltInAgentCapabilities;
};

/**
 * Direct, transport-free access to EditorAgentHarness for ARC's built-in AI.
 *
 * This adapter intentionally exposes only the same invoke/capability/event
 * surface available to transport clients. Approval, transaction, revision,
 * stable-id, and audit policy remain owned and enforced by the harness.
 */
export class BuiltInAgentAdapter {
  readonly clientId = BUILT_IN_AGENT_CLIENT_ID;
  readonly clientName = BUILT_IN_AGENT_CLIENT_NAME;
  private disposed = false;

  constructor(private readonly harness: EditorAgentHarness) {
    this.harness.touchClient(this.clientId, this.clientName);
  }

  async capabilities(): Promise<BuiltInAgentCapabilities> {
    return capabilitiesFromHarness(await this.invoke('agent.capabilities'));
  }

  async invoke(method: string, params: unknown = {}): Promise<unknown> {
    this.assertAvailable();
    if (typeof method !== 'string' || method.trim() === '') throw new Error('Built-in agent method is required');
    return this.harness.invoke(method, params, this.clientId);
  }

  onEvent(listener: (event: BuiltInAgentEvent) => void): () => void {
    this.assertAvailable();
    return this.harness.onEvent((event) => listener(event));
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.harness.disconnectClient(this.clientId);
  }

  private assertAvailable(): void {
    if (this.disposed) throw new Error('Built-in agent adapter is disposed');
  }
}
