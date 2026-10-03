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

export const providerToolName = (stableName: string): string =>
  `arc_${stableName
    .replaceAll('.', '_')
    .replaceAll(/([a-z0-9])([A-Z])/gu, '$1_$2')
    .replaceAll(/[^a-zA-Z0-9_-]/gu, '_')
    .toLocaleLowerCase()}`;

export const stableToolNameFromProvider = (providerName: string, tools: readonly AiToolDefinition[]): string =>
  tools.find((tool) => providerToolName(tool.name) === providerName)?.name ?? providerName;

export const projectAiToolsForOpenAi = (tools: readonly AiToolDefinition[]): OpenAiToolDefinition[] =>
  tools.map((tool) => ({
    type: 'function',
    name: providerToolName(tool.name),
    description: tool.description,
    parameters: tool.inputSchema,
  }));

export const projectAiToolsForAnthropic = (tools: readonly AiToolDefinition[]): AnthropicToolDefinition[] =>
  tools.map((tool) => ({
    name: providerToolName(tool.name),
    description: tool.description,
    input_schema: tool.inputSchema,
  }));
