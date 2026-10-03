import type { AiJsonObject, AiToolDefinition } from './aiRuntimeTypes';

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

export type BuiltInAgentToolInvokeRequest = Readonly<{
  name: string;
  arguments?: AiJsonObject;
}>;

export type BuiltInAgentToolExecutionResult = Readonly<{
  name: string;
  operation: string;
  content: string;
  truncated: boolean;
  originalBytes: number;
}>;

export type BuiltInAgentRuntimeBridge = {
  capabilities(): Promise<BuiltInAgentCapabilities>;
  invoke(method: string, params?: unknown): Promise<unknown>;
  tools(): Promise<AiToolDefinition[]>;
  invokeTool(name: string, arguments_?: AiJsonObject): Promise<BuiltInAgentToolExecutionResult>;
  onEvent(callback: (event: BuiltInAgentEvent) => void): () => void;
};
