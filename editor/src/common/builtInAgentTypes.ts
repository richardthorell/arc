export const BUILT_IN_AGENT_CLIENT_ID = 'arc.builtin-ai' as const;
export const BUILT_IN_AGENT_CLIENT_NAME = 'ARC Built-in AI' as const;

export type BuiltInAgentCapabilities = Readonly<{
  operations: readonly string[];
  editActions: readonly string[];
  [key: string]: unknown;
}>;

export type BuiltInAgentEvent = Readonly<{
  sequence: number;
  timestamp: string;
  type: string;
  entity?: Readonly<{ index: number; generation: number }>;
  message: string;
  payload: unknown;
}>;

export type BuiltInAgentInvokeRequest = Readonly<{
  method: string;
  params?: unknown;
}>;

export type BuiltInAgentRuntimeBridge = {
  capabilities(): Promise<BuiltInAgentCapabilities>;
  invoke(method: string, params?: unknown): Promise<unknown>;
  onEvent(callback: (event: BuiltInAgentEvent) => void): () => void;
};
