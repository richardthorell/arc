import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';

import { SceneGatewayCore, type GatewayHostResponse, type GatewayHostTransport } from './aiGatewayCore';
import { AiGatewayServer } from './aiGatewayServer';

const temporaryDirectories: string[] = [];
afterEach(() => {
  for (const directory of temporaryDirectories.splice(0))
    rmSync(directory, { recursive: true, force: true });
});

const reply = (payload: unknown = {}): GatewayHostResponse => ({
  kind: 'response', requestId: 1, succeeded: true, error: '', payload,
  sceneRevision: 3, worldEpoch: 1, frameRevision: 2,
});

class ServerHost implements GatewayHostTransport {
  async command(): Promise<GatewayHostResponse> { return reply(); }
  async query(type: string): Promise<GatewayHostResponse> {
    return reply(type === 'gateway.sceneEntities' ? { entities: [] } : {});
  }
}

describe('AiGatewayServer security adapters', () => {
  it('requires the launch token and rejects unapproved browser origins and save methods', async () => {
    const directory = mkdtempSync(path.join(os.tmpdir(), 'arc-ai-gateway-'));
    temporaryDirectories.push(directory);
    const core = new SceneGatewayCore(new ServerHost());
    const server = new AiGatewayServer(core, { appDataPath: directory });
    await server.start();
    try {
      const endpoint = core.status().endpoint;
      expect((await fetch(`${endpoint}/api/v1/status`)).status).toBe(401);
      expect((await fetch(`${endpoint}/api/v1/status`, {
        headers: { authorization: `Bearer ${core.token}`, origin: 'https://attacker.invalid' },
      })).status).toBe(403);
      expect((await fetch(`${endpoint}/api/v1/status`, {
        headers: { authorization: `Bearer ${core.token}` },
      })).status).toBe(200);

      const save = await fetch(`${endpoint}/rpc/v1`, {
        method: 'POST',
        headers: {
          authorization: `Bearer ${core.token}`,
          'content-type': 'application/json',
          'x-arc-client-id': 'test-client',
        },
        body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'scene.save', params: {} }),
      });
      const saveBody = await save.json() as { error?: { message?: string } };
      expect(saveBody.error?.message).toMatch(/Unsupported gateway method/);
    } finally {
      await server.stop();
    }
  });

  it('rotates the bearer token for every launch', () => {
    const first = new SceneGatewayCore(new ServerHost());
    const second = new SceneGatewayCore(new ServerHost());
    expect(first.token).toHaveLength(43);
    expect(second.token).toHaveLength(43);
    expect(first.token).not.toBe(second.token);
  });

  it('serves equivalent MCP and direct OpenAPI operations', async () => {
    const directory = mkdtempSync(path.join(os.tmpdir(), 'arc-ai-gateway-'));
    temporaryDirectories.push(directory);
    const core = new SceneGatewayCore(new ServerHost());
    const server = new AiGatewayServer(core, { appDataPath: directory });
    await server.start();
    const endpoint = core.status().endpoint;
    const headers = {
      authorization: `Bearer ${core.token}`,
      'x-arc-client-id': 'adapter-test',
    };
    const client = new Client({ name: 'arc-gateway-test', version: '1.0.0' });
    const transport = new StreamableHTTPClientTransport(new URL(`${endpoint}/mcp`), {
      requestInit: { headers },
      fetch: async (input, init) => {
        const response = await fetch(input, init);
        if (!response.ok)
          throw new Error(`MCP HTTP ${response.status}: ${await response.clone().text()}`);
        return response;
      },
    });
    try {
      await client.connect(transport);
      const tools = await client.listTools();
      expect(tools.tools.some((tool) => tool.name === 'arc_scene_overview')).toBe(true);
      expect(tools.tools.some((tool) => tool.name === 'arc_list_assets')).toBe(true);
      expect(tools.tools.some((tool) => tool.name === 'arc_history')).toBe(true);
      expect(tools.tools.some((tool) => tool.name === 'arc_debug_viewport')).toBe(true);
      expect(tools.tools.some((tool) => tool.name === 'arc_inspect_viewport_pixel')).toBe(true);
      expect(tools.tools.some((tool) => tool.name === 'arc_compare_viewport_captures')).toBe(true);
      const mcpResult = await client.callTool({ name: 'arc_scene_overview', arguments: {} });
      expect(mcpResult.isError).not.toBe(true);

      const direct = await fetch(`${endpoint}/api/v1/scene/overview`, {
        method: 'POST',
        headers: { ...headers, 'content-type': 'application/json' },
        body: '{}',
      });
      expect(direct.status).toBe(200);
      const directBody = await direct.json() as { result?: { entities?: unknown[] } };
      expect(directBody.result?.entities).toEqual([]);

      const openApi = await fetch(`${endpoint}/openapi.json`, { headers });
      const document = await openApi.json() as { paths?: Record<string, unknown> };
      expect(document.paths?.['/api/v1/scene/overview']).toBeDefined();
    } finally {
      await client.close();
      await server.stop();
    }
  });
});
