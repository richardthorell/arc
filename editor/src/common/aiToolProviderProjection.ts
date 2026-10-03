import type { AiJsonObject, AiToolDefinition } from './aiRuntimeTypes';

export type OpenAiToolDefinition = Readonly<{
  type: 'function';
  name: string;
  description: string;
  parameters: AiJsonObject;
}>;

export type AnthropicToolDefinition = Readonly<{
  name: string;
  description: string;
  input_schema: AiJsonObject;
}>;

export const projectAiToolsForOpenAi = (tools: readonly AiToolDefinition[]): OpenAiToolDefinition[] =>
  tools.map((tool) => ({
    type: 'function',
    name: tool.name,
    description: tool.description,
    parameters: tool.inputSchema,
  }));

export const projectAiToolsForAnthropic = (tools: readonly AiToolDefinition[]): AnthropicToolDefinition[] =>
  tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: tool.inputSchema,
  }));
