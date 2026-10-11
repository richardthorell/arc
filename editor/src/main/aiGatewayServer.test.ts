import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { z } from 'zod';

import { EditorAgentHarness, type AgentHarnessHost, type AgentHostResponse } from './editorAgentHarness';
import { AiGatewayServer } from './aiGatewayServer';

const temporaryDirectories: string[] = [];
afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

const reply = (payload: unknown = {}): AgentHostResponse => ({
  kind: 'response',
  requestId: 1,
  succeeded: true,
  error: '',
  payload,
  sceneRevision: 3,
  worldEpoch: 1,
  frameRevision: 2,
});

class ServerHost implements AgentHarnessHost {
  async command(): Promise<AgentHostResponse> {
    return reply();
  }
  async query(type: string): Promise<AgentHostResponse> {
    return reply(type === 'gateway.sceneEntities' ? { entities: [] } : {});
  }
}

describe('AiGatewayServer security adapters', () => {
  it('requires the launch token and rejects unapproved browser origins and save methods', async () => {
    const directory = mkdtempSync(path.join(os.tmpdir(), 'arc-ai-gateway-'));
    temporaryDirectories.push(directory);
    const core = new EditorAgentHarness(new ServerHost());
    const server = new AiGatewayServer(core, { appDataPath: directory });
    await server.start();
    try {
      const endpoint = server.status().endpoint;
      expect((await fetch(`${endpoint}/api/v1/status`)).status).toBe(401);
      expect(
        (
          await fetch(`${endpoint}/api/v1/status`, {
            headers: { authorization: `Bearer ${server.token}`, origin: 'https://attacker.invalid' },
          })
        ).status,
      ).toBe(403);
      expect(
        (
          await fetch(`${endpoint}/api/v1/status`, {
            headers: { authorization: `Bearer ${server.token}` },
          })
        ).status,
      ).toBe(200);

      const save = await fetch(`${endpoint}/rpc/v1`, {
        method: 'POST',
        headers: {
          authorization: `Bearer ${server.token}`,
          'content-type': 'application/json',
          'x-arc-client-id': 'test-client',
        },
        body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'scene.save', params: {} }),
      });
      const saveBody = (await save.json()) as { error?: { message?: string } };
      expect(saveBody.error?.message).toMatch(/Unsupported agent harness method/);
    } finally {
      await server.stop();
    }
  });

  it('rotates the bearer token for every launch', () => {
    const firstDirectory = mkdtempSync(path.join(os.tmpdir(), 'arc-ai-gateway-'));
    const secondDirectory = mkdtempSync(path.join(os.tmpdir(), 'arc-ai-gateway-'));
    temporaryDirectories.push(firstDirectory, secondDirectory);
    const first = new AiGatewayServer(new EditorAgentHarness(new ServerHost()), {
      appDataPath: firstDirectory,
    });
    const second = new AiGatewayServer(new EditorAgentHarness(new ServerHost()), {
      appDataPath: secondDirectory,
    });
    expect(first.token).toHaveLength(43);
    expect(second.token).toHaveLength(43);
    expect(first.token).not.toBe(second.token);
  });

  it('serves equivalent MCP and direct OpenAPI operations', async () => {
    const directory = mkdtempSync(path.join(os.tmpdir(), 'arc-ai-gateway-'));
    temporaryDirectories.push(directory);
    const core = new EditorAgentHarness(new ServerHost());
    core.registerOperation({
      id: 'test.inspect',
      version: 1,
      domain: 'test',
      description: 'Test domain discovery',
      permissions: ['editor.read'],
      requiredCapabilities: [],
      effects: { reads: ['test'], writes: [] },
      concurrency: 'independent-read',
      cache: 'revision',
      transaction: 'none',
      references: [],
      creates: [],
      input: z.object({ value: z.number() }).strict(),
      output: z.object({ doubled: z.number() }).strict(),
      execute: ({ value }) => ({ doubled: value * 2 }),
    });
    const server = new AiGatewayServer(core, { appDataPath: directory });
    await server.start();
    const endpoint = server.status().endpoint;
    const headers = {
      authorization: `Bearer ${server.token}`,
      'x-arc-client-id': 'adapter-test',
    };
    const client = new Client({ name: 'arc-gateway-test', version: '1.0.0' });
    const transport = new StreamableHTTPClientTransport(new URL(`${endpoint}/mcp`), {
      requestInit: { headers },
      fetch: async (input, init) => {
        const response = await fetch(input, init);
        if (!response.ok) throw new Error(`MCP HTTP ${response.status}: ${await response.clone().text()}`);
        return response;
      },
    });
    try {
      await client.connect(transport);
      const tools = await client.listTools();
      expect(tools.tools.some((tool) => tool.name === 'arc_agent_capabilities')).toBe(true);
      expect(tools.tools.some((tool) => tool.name === 'arc_scene_overview')).toBe(true);
      expect(tools.tools.some((tool) => tool.name === 'arc_list_assets')).toBe(true);
      expect(tools.tools.some((tool) => tool.name === 'arc_history')).toBe(true);
      expect(tools.tools.some((tool) => tool.name === 'arc_debug_viewport')).toBe(true);
      expect(tools.tools.some((tool) => tool.name === 'arc_inspect_viewport_pixel')).toBe(true);
      expect(tools.tools.some((tool) => tool.name === 'arc_compare_viewport_captures')).toBe(true);
      expect(tools.tools.find((tool) => tool.name === 'arc_test_inspect')?.inputSchema).toMatchObject({
        properties: { value: { type: 'number' } },
        required: ['value'],
        additionalProperties: false,
      });
      const extension = await client.callTool({ name: 'arc_test_inspect', arguments: { value: 4 } });
      expect(extension.structuredContent).toEqual({ doubled: 8 });
      const httpExtension = await fetch(`${endpoint}/api/v1/operations/test.inspect`, {
        method: 'POST',
        headers: { ...headers, 'content-type': 'application/json' },
        body: JSON.stringify({ value: 4 }),
      });
      expect(await httpExtension.json()).toMatchObject({ result: { doubled: 8 } });
      const mcpResult = await client.callTool({ name: 'arc_scene_overview', arguments: {} });
      expect(mcpResult.isError).not.toBe(true);

      const direct = await fetch(`${endpoint}/api/v1/scene/overview`, {
        method: 'POST',
        headers: { ...headers, 'content-type': 'application/json' },
        body: '{}',
      });
      expect(direct.status).toBe(200);
      const directBody = (await direct.json()) as { result?: { entities?: unknown[] } };
      expect(directBody.result?.entities).toEqual([]);

      const openApi = await fetch(`${endpoint}/openapi.json`, { headers });
      const document = (await openApi.json()) as { paths?: Record<string, unknown> };
      expect(document.paths?.['/api/v1/scene/overview']).toBeDefined();
      expect(document.paths?.['/api/v1/operations/test.inspect']).toMatchObject({
        post: { requestBody: { content: { 'application/json': { schema: { required: ['value'] } } } } },
      });
    } finally {
      await client.close();
      await server.stop();
    }
  });
});
