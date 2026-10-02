import type { AiRuntimeRequest } from './aiRuntimeTypes';

export const AI_SECURITY_POLICY_VERSION = 1 as const;
export const AI_REDACTED_VALUE = '[REDACTED]' as const;

export type AiDataOrigin = 'user' | 'conversation' | 'project' | 'editor' | 'skill' | 'tool';
export type AiDataSensitivity = 'standard' | 'sensitive' | 'secret' | 'credential';

export type AiOutboundDataItem = {
  id: string;
  label: string;
  origin: AiDataOrigin;
  sensitivity: AiDataSensitivity;
  projectScoped?: boolean;
  projectGuid?: string;
  inspectable?: boolean;
  explicitlyApproved?: boolean;
};

export type AiSecurityDecisionCode =
  | 'allowed'
  | 'cancelled'
  | 'credential_blocked'
  | 'secret_blocked'
  | 'project_scope_missing'
  | 'project_scope_mismatch'
  | 'context_not_inspectable'
  | 'sensitive_context_requires_approval'
  | 'restricted_tool'
  | 'tool_not_in_harness'
  | 'tool_capability_missing'
  | 'mutation_requires_harness_approval'
  | 'unsafe_runtime_metadata';

export type AiSecurityDecision =
  | { allowed: true; code: 'allowed' }
  | { allowed: false; code: Exclude<AiSecurityDecisionCode, 'allowed'>; reason: string };

const allowed = (): AiSecurityDecision => ({ allowed: true, code: 'allowed' });
const denied = (code: Exclude<AiSecurityDecisionCode, 'allowed'>, reason: string): AiSecurityDecision => ({
  allowed: false,
  code,
  reason,
});

const inspectableOrigins = new Set<AiDataOrigin>(['project', 'editor', 'skill', 'tool']);

export const evaluateAiOutboundDataItem = (
  item: AiOutboundDataItem,
  activeProjectGuid?: string,
): AiSecurityDecision => {
  if (item.sensitivity === 'credential')
    return denied('credential_blocked', `${item.label} is a credential and cannot be sent to a model`);
  if (item.sensitivity === 'secret')
    return denied('secret_blocked', `${item.label} is secret data and cannot be sent to a model`);

  if (item.projectScoped) {
    if (!activeProjectGuid || !item.projectGuid)
      return denied('project_scope_missing', `${item.label} is project-scoped but has no active project identity`);
    if (item.projectGuid !== activeProjectGuid)
      return denied('project_scope_mismatch', `${item.label} belongs to a different project`);
  }

  if (inspectableOrigins.has(item.origin) && !item.inspectable)
    return denied('context_not_inspectable', `${item.label} must be inspectable before it can leave the editor`);

  if (item.sensitivity === 'sensitive' && !item.explicitlyApproved)
    return denied(
      'sensitive_context_requires_approval',
      `${item.label} is sensitive and requires explicit approval before it can be sent`,
    );

  return allowed();
};

export const assertAiOutboundDataAllowed = (
  items: readonly AiOutboundDataItem[],
  activeProjectGuid?: string,
): void => {
  for (const item of items) {
    const decision = evaluateAiOutboundDataItem(item, activeProjectGuid);
    if (!decision.allowed) throw new Error(`[AI security: ${decision.code}] ${decision.reason}`);
  }
};

export type AiRestrictedOperation =
  | 'arbitrary-process'
  | 'unrestricted-filesystem'
  | 'project-lifecycle'
  | 'build-package';

export type AiToolSecurityDescriptor = {
  name: string;
  boundary: 'harness' | 'restricted';
  harnessOperation?: string;
  mutating?: boolean;
  requiresHarnessApproval?: boolean;
  restriction?: AiRestrictedOperation;
};

export type AiToolExecutionSecurityContext = {
  harnessCapabilities: ReadonlySet<string>;
  signal?: AbortSignal;
};

export const evaluateAiToolInvocation = (
  tool: AiToolSecurityDescriptor,
  context: AiToolExecutionSecurityContext,
): AiSecurityDecision => {
  if (context.signal?.aborted) return denied('cancelled', 'AI operation was cancelled before tool execution');
  if (tool.boundary === 'restricted')
    return denied(
      'restricted_tool',
      `${tool.name} requests ${tool.restriction ?? 'a restricted operation'} outside EditorAgentHarness`,
    );
  if (!tool.harnessOperation)
    return denied('tool_not_in_harness', `${tool.name} is not mapped to an EditorAgentHarness operation`);
  if (!context.harnessCapabilities.has(tool.harnessOperation))
    return denied('tool_capability_missing', `${tool.harnessOperation} is not available from EditorAgentHarness`);
  if (tool.mutating && !tool.requiresHarnessApproval)
    return denied(
      'mutation_requires_harness_approval',
      `${tool.name} mutates editor state but does not preserve harness approval semantics`,
    );
  return allowed();
};

export const assertAiToolInvocationAllowed = (
  tool: AiToolSecurityDescriptor,
  context: AiToolExecutionSecurityContext,
): void => {
  const decision = evaluateAiToolInvocation(tool, context);
  if (!decision.allowed) throw new Error(`[AI security: ${decision.code}] ${decision.reason}`);
};

export const resolveAiSkillTools = (
  requestedToolNames: readonly string[],
  registeredTools: readonly AiToolSecurityDescriptor[],
): AiToolSecurityDescriptor[] => {
  const requested = new Set(requestedToolNames);
  return registeredTools.filter((tool) => requested.has(tool.name));
};

const normalizeFieldName = (value: string): string => value.replace(/[^a-z0-9]/gi, '').toLocaleLowerCase();

const secretFieldNames = new Set([
  'apikey',
  'authorization',
  'accesstoken',
  'refreshtoken',
  'password',
  'secret',
  'clientsecret',
  'credential',
  'credentials',
  'cookie',
  'setcookie',
  'xapikey',
]);

export const isAiSecretFieldName = (name: string): boolean => {
  const normalized = normalizeFieldName(name);
  return secretFieldNames.has(normalized) || normalized.endsWith('apikey') || normalized.endsWith('clientsecret');
};

const findSecretMetadataPath = (value: unknown, path = 'metadata'): string | null => {
  if (!value || typeof value !== 'object') return null;
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; ++index) {
      const nested = findSecretMetadataPath(value[index], `${path}[${index}]`);
      if (nested) return nested;
    }
    return null;
  }
  for (const [key, nestedValue] of Object.entries(value as Record<string, unknown>)) {
    const nestedPath = `${path}.${key}`;
    if (isAiSecretFieldName(key)) return nestedPath;
    const nested = findSecretMetadataPath(nestedValue, nestedPath);
    if (nested) return nested;
  }
  return null;
};

export const assertAiRuntimeRequestSafeForProvider = (request: AiRuntimeRequest): void => {
  const secretPath = findSecretMetadataPath(request.metadata);
  if (secretPath)
    throw new Error(
      `[AI security: unsafe_runtime_metadata] Provider request metadata contains secret-like field '${secretPath}'`,
    );
};

export const redactAiDiagnosticText = (value: string): string =>
  value
    .replace(/\bBearer\s+[A-Za-z0-9._~+/=-]{8,}/gi, `Bearer ${AI_REDACTED_VALUE}`)
    .replace(/\bsk-[A-Za-z0-9_-]{8,}\b/g, AI_REDACTED_VALUE)
    .replace(
      /\b(api[-_ ]?key|x-api-key|access[-_ ]?token|refresh[-_ ]?token|password|credential)\s*[:=]\s*([^\s,;]+)/gi,
      (_match, key: string) => `${key}=${AI_REDACTED_VALUE}`,
    );

export const redactAiDiagnosticValue = (value: unknown): unknown => {
  if (typeof value === 'string') return redactAiDiagnosticText(value);
  if (Array.isArray(value)) return value.map((entry) => redactAiDiagnosticValue(entry));
  if (!value || typeof value !== 'object') return value;

  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([key, nestedValue]) => [
      key,
      isAiSecretFieldName(key) ? AI_REDACTED_VALUE : redactAiDiagnosticValue(nestedValue),
    ]),
  );
};

export type AiRuntimeDiagnosticSummary = {
  conversationId: string;
  messageCount: number;
  messageRoles: Record<string, number>;
  toolNames: string[];
  metadataKeys: string[];
  cancelled: boolean;
};

export const summarizeAiRuntimeRequestForDiagnostics = (request: AiRuntimeRequest): AiRuntimeDiagnosticSummary => {
  const messageRoles: Record<string, number> = {};
  for (const message of request.messages) messageRoles[message.role] = (messageRoles[message.role] ?? 0) + 1;
  return {
    conversationId: request.conversationId,
    messageCount: request.messages.length,
    messageRoles,
    toolNames: request.tools?.map((tool) => tool.name) ?? [],
    metadataKeys: request.metadata
      ? Object.keys(request.metadata).filter((key) => !isAiSecretFieldName(key)).sort()
      : [],
    cancelled: request.signal?.aborted ?? false,
  };
};
