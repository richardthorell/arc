import { describe, expect, it } from 'vitest';
import { textContent, type AiRuntimeRequest } from './aiRuntimeTypes';
import {
  AI_REDACTED_VALUE,
  assertAiRuntimeRequestSafeForProvider,
  evaluateAiOutboundDataItem,
  evaluateAiToolInvocation,
  redactAiDiagnosticText,
  redactAiDiagnosticValue,
  resolveAiSkillTools,
  summarizeAiRuntimeRequestForDiagnostics,
  type AiToolSecurityDescriptor,
} from './aiSecurityPolicy';

describe('AI security policy', () => {
  it('never allows credentials or secrets to leave the editor context boundary', () => {
    expect(
      evaluateAiOutboundDataItem({
        id: 'credential',
        label: 'OpenAI API key',
        origin: 'editor',
        sensitivity: 'credential',
        inspectable: true,
        explicitlyApproved: true,
      }),
    ).toMatchObject({ allowed: false, code: 'credential_blocked' });

    expect(
      evaluateAiOutboundDataItem({
        id: 'secret',
        label: 'Signing secret',
        origin: 'project',
        sensitivity: 'secret',
        projectScoped: true,
        projectGuid: 'project-a',
        inspectable: true,
        explicitlyApproved: true,
      }, 'project-a'),
    ).toMatchObject({ allowed: false, code: 'secret_blocked' });
  });

  it('requires project context to match the active project and remain inspectable', () => {
    const context = {
      id: 'selection',
      label: 'Current selection',
      origin: 'editor' as const,
      sensitivity: 'standard' as const,
      projectScoped: true,
      projectGuid: 'project-a',
      inspectable: true,
    };

    expect(evaluateAiOutboundDataItem(context, 'project-a')).toEqual({ allowed: true, code: 'allowed' });
    expect(evaluateAiOutboundDataItem(context, 'project-b')).toMatchObject({
      allowed: false,
      code: 'project_scope_mismatch',
    });
    expect(evaluateAiOutboundDataItem({ ...context, inspectable: false }, 'project-a')).toMatchObject({
      allowed: false,
      code: 'context_not_inspectable',
    });
  });

  it('requires explicit approval for sensitive context', () => {
    const sensitiveContext = {
      id: 'diagnostic',
      label: 'Sensitive diagnostic',
      origin: 'editor' as const,
      sensitivity: 'sensitive' as const,
      inspectable: true,
    };

    expect(evaluateAiOutboundDataItem(sensitiveContext)).toMatchObject({
      allowed: false,
      code: 'sensitive_context_requires_approval',
    });
    expect(evaluateAiOutboundDataItem({ ...sensitiveContext, explicitlyApproved: true })).toEqual({
      allowed: true,
      code: 'allowed',
    });
  });

  it('lets skills select registered tools but never synthesize new authority', () => {
    const registeredTools: AiToolSecurityDescriptor[] = [
      {
        name: 'scene.inspect',
        boundary: 'harness',
        harnessOperation: 'scene.inspect',
      },
      {
        name: 'scene.rename',
        boundary: 'harness',
        harnessOperation: 'entity.rename',
        mutating: true,
        requiresHarnessApproval: true,
      },
    ];

    expect(resolveAiSkillTools(['scene.rename', 'process.exec'], registeredTools)).toEqual([registeredTools[1]]);
  });

  it('keeps restricted operations outside the harness and preserves mutation approval semantics', () => {
    const capabilities = new Set(['scene.inspect', 'entity.rename']);
    expect(
      evaluateAiToolInvocation(
        { name: 'process.exec', boundary: 'restricted', restriction: 'arbitrary-process' },
        { harnessCapabilities: capabilities },
      ),
    ).toMatchObject({ allowed: false, code: 'restricted_tool' });

    expect(
      evaluateAiToolInvocation(
        {
          name: 'scene.rename',
          boundary: 'harness',
          harnessOperation: 'entity.rename',
          mutating: true,
        },
        { harnessCapabilities: capabilities },
      ),
    ).toMatchObject({ allowed: false, code: 'mutation_requires_harness_approval' });

    expect(
      evaluateAiToolInvocation(
        {
          name: 'scene.rename',
          boundary: 'harness',
          harnessOperation: 'entity.rename',
          mutating: true,
          requiresHarnessApproval: true,
        },
        { harnessCapabilities: capabilities },
      ),
    ).toEqual({ allowed: true, code: 'allowed' });
  });

  it('blocks tool execution after cancellation', () => {
    const controller = new AbortController();
    controller.abort();
    expect(
      evaluateAiToolInvocation(
        { name: 'scene.inspect', boundary: 'harness', harnessOperation: 'scene.inspect' },
        { harnessCapabilities: new Set(['scene.inspect']), signal: controller.signal },
      ),
    ).toMatchObject({ allowed: false, code: 'cancelled' });
  });

  it('rejects secret-like runtime metadata before provider dispatch', () => {
    const safeRequest: AiRuntimeRequest = {
      conversationId: 'conversation',
      messages: [{ id: 'message', role: 'user', content: [textContent('Hello')] }],
      metadata: { projectGuid: 'project-a', contextCount: 2 },
    };
    expect(() => assertAiRuntimeRequestSafeForProvider(safeRequest)).not.toThrow();

    expect(() =>
      assertAiRuntimeRequestSafeForProvider({
        ...safeRequest,
        metadata: { projectGuid: 'project-a', nested: { openaiApiKey: 'sk-not-for-provider-payload' } },
      }),
    ).toThrow(/unsafe_runtime_metadata/);
  });

  it('redacts credentials from diagnostics without retaining prompt or metadata values in summaries', () => {
    expect(redactAiDiagnosticText('Authorization: Bearer abcdefghijklmnop')).toBe(
      `Authorization: Bearer ${AI_REDACTED_VALUE}`,
    );
    expect(redactAiDiagnosticText('api_key=sk-abcdefghijklmnopqrstuvwxyz')).toBe(
      `api_key=${AI_REDACTED_VALUE}`,
    );
    expect(
      redactAiDiagnosticValue({
        provider: 'openai',
        apiKey: 'sk-abcdefghijklmnopqrstuvwxyz',
        nested: { authorization: 'Bearer abcdefghijklmnop', status: 429 },
      }),
    ).toEqual({
      provider: 'openai',
      apiKey: AI_REDACTED_VALUE,
      nested: { authorization: AI_REDACTED_VALUE, status: 429 },
    });

    const request: AiRuntimeRequest = {
      conversationId: 'conversation',
      messages: [
        { id: 'message', role: 'user', content: [textContent('private prompt contents')] },
        { id: 'assistant', role: 'assistant', content: 'private response contents' },
      ],
      tools: [{ name: 'scene.inspect', description: 'Inspect scene', inputSchema: {} }],
      metadata: { projectGuid: 'project-a', apiKey: 'must-not-appear' },
    };
    const summary = summarizeAiRuntimeRequestForDiagnostics(request);
    const serialized = JSON.stringify(summary);

    expect(summary).toEqual({
      conversationId: 'conversation',
      messageCount: 2,
      messageRoles: { user: 1, assistant: 1 },
      toolNames: ['scene.inspect'],
      metadataKeys: ['projectGuid'],
      cancelled: false,
    });
    expect(serialized).not.toContain('private prompt contents');
    expect(serialized).not.toContain('private response contents');
    expect(serialized).not.toContain('must-not-appear');
  });
});
