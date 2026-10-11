import { AsyncLocalStorage } from 'node:async_hooks';
import { randomBytes, randomUUID, timingSafeEqual } from 'node:crypto';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { chmodSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { gzipSync } from 'node:zlib';
import { nativeImage } from 'electron';
import { McpServer, ResourceTemplate } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { z } from 'zod';
import { EditorAgentHarness, type AgentHarnessStatus } from './editorAgentHarness';
import { gatewayHttpMethods } from './agentHarnessContract';

import { operationMcpName } from './agentOperationCatalog';
import { EditorOperationError } from '../common/editorOperationRegistry';

type RequestContext = { clientId: string };
export type GatewayStatus = AgentHarnessStatus & {
  enabled: boolean;
  endpoint: string;
  discoveryFile: string;
  protocolVersion: 1;
};

type GatewayServerOptions = {
  appDataPath: string;
  onStatus?: (status: GatewayStatus) => void;
};

const maximumRequestBytes = 1024 * 1024;
const requestsPerMinute = 120;
const maximumArtifactBytes = 256 * 1024 * 1024;
const artifactLifetimeMilliseconds = 10 * 60 * 1000;
type GatewayArtifact = {
  id: string;
  mimeType: string;
  data: Buffer;
  createdAt: number;
  lastAccessAt: number;
};

const readBody = async (request: IncomingMessage): Promise<unknown> => {
  const chunks: Buffer[] = [];
  let length = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    length += buffer.length;
    if (length > maximumRequestBytes) throw new Error('Request body exceeds 1 MiB');
    chunks.push(buffer);
  }
  if (length === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown;
};

const sendJson = (response: ServerResponse, status: number, value: unknown): void => {
  const body = JSON.stringify(value);
  response.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
    'cache-control': 'no-store',
  });
  response.end(body);
};

const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export class AiGatewayServer {
  readonly token = randomBytes(32).toString('base64url');
  private server: Server | null = null;
  private endpoint = '';
  private readonly discoveryPath: string;
  private readonly context = new AsyncLocalStorage<RequestContext>();
  private readonly rateLimits = new Map<string, { started: number; count: number }>();
  private readonly eventStreams = new Set<ServerResponse>();
  private readonly artifacts = new Map<string, GatewayArtifact>();
  private unsubscribeStatus?: () => void;
  private unsubscribeEvents?: () => void;
  private authorityTimer?: NodeJS.Timeout;

  constructor(
    readonly harness: EditorAgentHarness,
    private readonly options: GatewayServerOptions,
  ) {
    const directory = path.join(options.appDataPath, 'ai-gateway');
    this.discoveryPath = path.join(directory, 'active.json');
    mkdirSync(directory, { recursive: true, mode: 0o700 });
  }

  status(): GatewayStatus {
    return {
      ...this.harness.status(),
      enabled: Boolean(this.endpoint),
      endpoint: this.endpoint,
      discoveryFile: this.discoveryPath,
      protocolVersion: 1,
    };
  }

  private publicStatus(): Omit<GatewayStatus, 'audit' | 'discoveryFile'> {
    const status = this.status();
    return {
      enabled: status.enabled,
      endpoint: status.endpoint,
      protocolVersion: status.protocolVersion,
      sceneRevision: status.sceneRevision,
      worldEpoch: status.worldEpoch,
      frameRevision: status.frameRevision,
      eventSequence: status.eventSequence,
      clients: status.clients,
      pendingEditRequests: status.pendingEditRequests,
      activeEditSession: status.activeEditSession,
      lastCommittedEdit: status.lastCommittedEdit,
      viewportLease: status.viewportLease,
    };
  }

  async start(): Promise<void> {
    if (this.server) return;
    this.server = createServer((request, response) => {
      void this.handle(request, response);
    });
    await new Promise<void>((resolve, reject) => {
      this.server?.once('error', reject);
      this.server?.listen(0, '127.0.0.1', () => resolve());
    });
    const address = this.server.address();
    if (!address || typeof address === 'string') throw new Error('AI gateway did not receive a TCP port');
    this.endpoint = `http://127.0.0.1:${address.port}`;
    const discovery = {
      protocolVersion: 1,
      endpoint: this.endpoint,
      mcpEndpoint: `${this.endpoint}/mcp`,
      rpcEndpoint: `${this.endpoint}/rpc/v1`,
      openApiEndpoint: `${this.endpoint}/openapi.json`,
      token: this.token,
      pid: process.pid,
      startedAt: new Date().toISOString(),
    };
    writeFileSync(this.discoveryPath, JSON.stringify(discovery, null, 2), { encoding: 'utf8', mode: 0o600 });
    try {
      chmodSync(this.discoveryPath, 0o600);
    } catch {
      // Windows applies the containing user-profile ACL; chmod is best-effort.
    }
    this.unsubscribeStatus = this.harness.onStatus(() => {
      const status = this.status();
      this.options.onStatus?.(status);
      this.publishEvent('gateway.status', status);
    });
    this.unsubscribeEvents = this.harness.onEvent((event) => this.publishEvent('arc.event', event));
    this.authorityTimer = setInterval(() => {
      void this.harness.expireInactiveAuthority();
    }, 30_000);
    this.authorityTimer.unref();
    this.options.onStatus?.(this.status());
  }

  async stop(): Promise<void> {
    if (this.authorityTimer) {
      clearInterval(this.authorityTimer);
      this.authorityTimer = undefined;
    }
    await this.harness.invalidateAuthority('editor shutdown');
    this.unsubscribeStatus?.();
    this.unsubscribeStatus = undefined;
    this.unsubscribeEvents?.();
    this.unsubscribeEvents = undefined;
    for (const stream of this.eventStreams) stream.end();
    this.eventStreams.clear();
    rmSync(this.discoveryPath, { force: true });
    if (this.server) {
      const server = this.server;
      this.server = null;
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  }

  private async handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
    try {
      if (!this.authorized(request)) {
        sendJson(response, 401, { error: 'Missing or invalid ARC gateway token' });
        return;
      }
      if (!this.validHost(request)) {
        sendJson(response, 403, { error: 'Host header is not the active localhost gateway' });
        return;
      }
      if (!this.validOrigin(request)) {
        sendJson(response, 403, { error: 'Origin is not allowed' });
        return;
      }
      const clientId = this.clientId(request);
      if (!this.withinRateLimit(clientId)) {
        sendJson(response, 429, { error: 'Gateway rate limit exceeded' });
        return;
      }
      this.harness.touchClient(clientId, this.clientName(request));
      const url = new URL(request.url ?? '/', this.endpoint);

      if (url.pathname === '/mcp') {
        if (request.method !== 'POST') {
          sendJson(response, 405, {
            jsonrpc: '2.0',
            error: { code: -32000, message: 'Stateless MCP accepts POST requests only' },
            id: null,
          });
          return;
        }
        const { server, transport } = this.createMcp();
        await server.connect(transport);
        response.once('close', () => {
          void transport.close();
          void server.close();
        });
        await this.context.run({ clientId }, () => transport.handleRequest(request, response));
        return;
      }
      if (url.pathname === '/events' && request.method === 'GET') {
        response.writeHead(200, {
          'content-type': 'text/event-stream',
          'cache-control': 'no-cache',
          connection: 'keep-alive',
        });
        response.write(`event: gateway.status\ndata: ${JSON.stringify(this.status())}\n\n`);
        this.eventStreams.add(response);
        request.on('close', () => {
          this.eventStreams.delete(response);
          void this.harness.disconnectClient(clientId);
        });
        return;
      }
      if (url.pathname === '/openapi.json' && request.method === 'GET') {
        sendJson(response, 200, this.openApi());
        return;
      }
      if (url.pathname.startsWith('/artifacts/') && request.method === 'GET') {
        const artifact = this.getArtifact(url.pathname.slice('/artifacts/'.length));
        if (!artifact) {
          sendJson(response, 404, { error: 'Capture artifact was not found or has expired' });
          return;
        }
        response.writeHead(200, {
          'content-type': artifact.mimeType,
          'content-length': artifact.data.length,
          'cache-control': 'private, no-store',
        });
        response.end(artifact.data);
        return;
      }
      if (url.pathname === '/api/v1/status' && request.method === 'GET') {
        sendJson(response, 200, this.publicStatus());
        return;
      }
      if (url.pathname === '/api/v1/invoke' && request.method === 'POST') {
        const body = (await readBody(request)) as { method?: unknown; params?: unknown };
        if (typeof body.method !== 'string') throw new Error('method is required');
        sendJson(response, 200, { result: await this.invoke(body.method, body.params, clientId) });
        return;
      }
      const apiMethod = this.httpOperations()[url.pathname];
      if (apiMethod && request.method === 'POST') {
        const params = await readBody(request);
        sendJson(response, 200, { result: await this.invoke(apiMethod, params, clientId) });
        return;
      }
      if (url.pathname === '/rpc/v1' && request.method === 'POST') {
        const body = (await readBody(request)) as {
          jsonrpc?: unknown;
          id?: unknown;
          method?: unknown;
          params?: unknown;
        };
        if (body.jsonrpc !== '2.0' || typeof body.method !== 'string') {
          sendJson(response, 400, {
            jsonrpc: '2.0',
            id: body.id ?? null,
            error: { code: -32600, message: 'Invalid JSON-RPC request' },
          });
          return;
        }
        try {
          const result = await this.invoke(body.method, body.params, clientId);
          sendJson(response, 200, { jsonrpc: '2.0', id: body.id ?? null, result });
        } catch (error) {
          sendJson(response, 200, {
            jsonrpc: '2.0',
            id: body.id ?? null,
            error: {
              code: -32000,
              message: errorMessage(error),
              ...(error instanceof EditorOperationError
                ? { data: { code: error.code, operation: error.operation } }
                : {}),
            },
          });
        }
        return;
      }
      sendJson(response, 404, { error: 'Gateway endpoint not found' });
    } catch (error) {
      sendJson(response, 400, {
        error: errorMessage(error),
        ...(error instanceof EditorOperationError ? { code: error.code, operation: error.operation } : {}),
      });
    }
  }

  private createMcp(): {
    server: McpServer;
    transport: StreamableHTTPServerTransport;
  } {
    const server = new McpServer({ name: 'arc-editor', version: '1.0.0' });
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
    const result = (value: unknown) => {
      const content: Array<{ type: 'text'; text: string } | { type: 'image'; data: string; mimeType: string }> = [
        { type: 'text', text: JSON.stringify(value, null, 2) },
      ];
      const colorArtifact = this.findColorArtifact(value);
      if (colorArtifact) {
        content.unshift({
          type: 'image',
          data: colorArtifact.data.toString('base64'),
          mimeType: colorArtifact.mimeType,
        });
      }
      return {
        content,
        structuredContent: value && typeof value === 'object' ? (value as Record<string, unknown>) : { value },
      };
    };
    const invoke = async (method: string, params: unknown) => {
      try {
        return result(await this.invoke(method, params, this.currentClient()));
      } catch (error) {
        if (!(error instanceof EditorOperationError)) throw error;
        return {
          ...result({ error: { code: error.code, message: error.message, operation: error.operation } }),
          isError: true,
        };
      }
    };

    for (const { descriptor, input } of this.harness.operationTransportEntries(this.currentClient())) {
      if (!descriptor.available) continue;
      server.registerTool(
        operationMcpName(descriptor.id),
        {
          description: descriptor.description,
          inputSchema: input,
          annotations: { readOnlyHint: descriptor.effects.writes.length === 0, openWorldHint: false },
        },
        async (params) => invoke(descriptor.id, params),
      );
    }
    server.registerTool(
      'arc_finish_edit',
      {
        description: 'Commit or cancel an active AI edit transaction.',
        inputSchema: {
          editSessionId: z.string(),
          action: z.enum(['commit', 'cancel']),
          expectedSceneRevision: z.number().int().positive().optional(),
        },
        annotations: { destructiveHint: false, openWorldHint: false },
      },
      async ({ action, ...params }) =>
        invoke(
          action === 'commit' ? 'edit.commit' : 'edit.cancel',
          action === 'cancel' ? { editSessionId: params.editSessionId } : params,
        ),
    );
    server.registerTool(
      'arc_history',
      {
        description: 'Undo or redo one validated in-memory scene history operation after edit access is approved.',
        inputSchema: {
          action: z.enum(['undo', 'redo']),
          expectedSceneRevision: z.number().int().positive(),
        },
        annotations: { destructiveHint: true, openWorldHint: false },
      },
      async ({ action, ...params }) => invoke(action === 'undo' ? 'history.undo' : 'history.redo', params),
    );

    const resource = async (uri: URL, method: string, params: unknown = {}) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: 'application/json',
          text: JSON.stringify(await this.invoke(method, params, this.currentClient()), null, 2),
        },
      ],
    });
    server.registerResource(
      'scene-summary',
      'arc://scene/summary',
      { title: 'ARC scene summary', mimeType: 'application/json' },
      async (uri) => resource(uri, 'scene.overview'),
    );
    server.registerResource(
      'component-schemas',
      'arc://schema/components',
      { title: 'ARC component schemas', mimeType: 'application/json' },
      async (uri) => resource(uri, 'scene.componentSchemas'),
    );
    server.registerResource(
      'viewport-latest',
      'arc://viewport/latest',
      { title: 'ARC viewport state', mimeType: 'application/json' },
      async (uri) => resource(uri, 'viewport.state'),
    );
    server.registerResource(
      'diagnostics-latest',
      'arc://diagnostics/latest',
      { title: 'ARC diagnostics', mimeType: 'application/json' },
      async (uri) => resource(uri, 'diagnostics.get'),
    );
    server.registerResource(
      'scene-entity',
      new ResourceTemplate('arc://scene/entity/{guid}', { list: undefined }),
      { title: 'ARC scene entity', mimeType: 'application/json' },
      async (uri, variables) => resource(uri, 'scene.getEntity', { guid: String(variables.guid) }),
    );
    return { server, transport };
  }

  private currentClient(): string {
    return this.context.getStore()?.clientId ?? 'mcp-local';
  }

  private async invoke(method: string, params: unknown, clientId: string): Promise<unknown> {
    if (method === 'gateway.status') return this.publicStatus();
    return this.materializeCaptureArtifacts(await this.harness.invoke(method, params, clientId));
  }

  private materializeCaptureArtifacts(value: unknown, inheritedMaxWidth = 1920, inheritedMaxHeight = 1080): unknown {
    if (Array.isArray(value)) {
      return value.map((entry) => this.materializeCaptureArtifacts(entry, inheritedMaxWidth, inheritedMaxHeight));
    }
    if (!value || typeof value !== 'object') return value;
    const record = value as Record<string, unknown>;
    const maxWidth = Math.min(inheritedMaxWidth, Number(record.captureMaxWidth) || inheritedMaxWidth);
    const maxHeight = Math.min(inheritedMaxHeight, Number(record.captureMaxHeight) || inheritedMaxHeight);
    const output: Record<string, unknown> = {};
    for (const [key, nested] of Object.entries(record)) {
      if (key === 'images' && Array.isArray(nested)) {
        output.images = nested.map((item) => this.materializeCaptureImage(item, maxWidth, maxHeight));
      } else {
        output[key] = this.materializeCaptureArtifacts(nested, maxWidth, maxHeight);
      }
    }
    return output;
  }

  private materializeCaptureImage(value: unknown, maxWidth: number, maxHeight: number): unknown {
    if (!value || typeof value !== 'object') return value;
    const image = value as Record<string, unknown>;
    if (typeof image.data !== 'string') return image;
    const channel = String(image.channel ?? 'capture');
    const format = String(image.format ?? '');
    const width = Math.max(1, Number(image.width) || 1);
    const height = Math.max(1, Number(image.height) || 1);
    const raw = Buffer.from(image.data, 'base64');
    const scale = Math.min(1, maxWidth / width, maxHeight / height);
    const artifactWidth = Math.max(1, Math.round(width * scale));
    const artifactHeight = Math.max(1, Math.round(height * scale));
    const png = this.capturePng(raw, format, width, height, maxWidth, maxHeight);
    const pngArtifact = this.storeArtifact(png, 'image/png');
    const materialized: Record<string, unknown> = {
      channel,
      format,
      width,
      height,
      artifact: {
        id: pngArtifact.id,
        mimeType: pngArtifact.mimeType,
        width: artifactWidth,
        height: artifactHeight,
        url: `${this.endpoint}/artifacts/${pngArtifact.id}`,
      },
    };
    const compressed = this.storeArtifact(gzipSync(raw), 'application/gzip');
    const elementType =
      format === 'r32f'
        ? 'float32-le'
        : format === 'r32ui'
          ? 'uint32-le'
          : format === 'rgba16f'
            ? 'float16x4-le'
            : 'uint8x4';
    materialized.rawArtifact = {
      id: compressed.id,
      mimeType: compressed.mimeType,
      encoding: 'gzip',
      elementType,
      byteLength: raw.length,
      url: `${this.endpoint}/artifacts/${compressed.id}`,
    };
    return materialized;
  }

  private capturePng(
    raw: Buffer,
    format: string,
    width: number,
    height: number,
    maxWidth: number,
    maxHeight: number,
  ): Buffer {
    const pixels = width * height;
    const bgra = Buffer.allocUnsafe(pixels * 4);
    const setPixel = (index: number, red: number, green: number, blue: number, alpha = 255) => {
      const offset = index * 4;
      bgra[offset] = Math.max(0, Math.min(255, blue));
      bgra[offset + 1] = Math.max(0, Math.min(255, green));
      bgra[offset + 2] = Math.max(0, Math.min(255, red));
      bgra[offset + 3] = Math.max(0, Math.min(255, alpha));
    };
    if (format === 'bgra8') {
      raw.copy(bgra, 0, 0, Math.min(raw.length, bgra.length));
    } else if (format === 'rgba8') {
      for (let index = 0; index < pixels; ++index) {
        const offset = index * 4;
        setPixel(index, raw[offset] ?? 0, raw[offset + 1] ?? 0, raw[offset + 2] ?? 0, raw[offset + 3] ?? 255);
      }
    } else if (format === 'rgba16f') {
      for (let index = 0; index < pixels; ++index) {
        const offset = index * 8;
        setPixel(
          index,
          this.halfToByte(raw.readUInt16LE(offset)),
          this.halfToByte(raw.readUInt16LE(offset + 2)),
          this.halfToByte(raw.readUInt16LE(offset + 4)),
          this.halfToByte(raw.readUInt16LE(offset + 6)),
        );
      }
    } else if (format === 'r32ui') {
      for (let index = 0; index < pixels; ++index) {
        const id = raw.readUInt32LE(index * 4);
        const hash = Math.imul(id ^ (id >>> 16), 0x45d9f3b);
        setPixel(
          index,
          id === 0 ? 0 : hash & 255,
          id === 0 ? 0 : (hash >>> 8) & 255,
          id === 0 ? 0 : (hash >>> 16) & 255,
        );
      }
    } else if (format === 'r32f') {
      const depths = new Float32Array(raw.buffer, raw.byteOffset, Math.min(pixels, raw.length / 4));
      let maximum = 0;
      for (const depth of depths) if (Number.isFinite(depth)) maximum = Math.max(maximum, depth);
      const denominator = Math.log2(1 + Math.max(maximum, 1));
      for (let index = 0; index < pixels; ++index) {
        const depth = depths[index] ?? 0;
        const value = Number.isFinite(depth) ? Math.round((255 * Math.log2(1 + Math.max(0, depth))) / denominator) : 0;
        setPixel(index, value, value, value);
      }
    } else {
      bgra.fill(0);
    }
    let image = nativeImage.createFromBitmap(bgra, { width, height, scaleFactor: 1 });
    const scale = Math.min(1, maxWidth / width, maxHeight / height);
    if (scale < 1) {
      image = image.resize({
        width: Math.max(1, Math.round(width * scale)),
        height: Math.max(1, Math.round(height * scale)),
        quality: 'best',
      });
    }
    return image.toPNG();
  }

  private halfToByte(bits: number): number {
    const sign = bits >>> 15;
    const exponent = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    let value: number;
    if (exponent === 0) value = mantissa * 2 ** -24;
    else if (exponent === 31) value = mantissa ? 0 : Number.POSITIVE_INFINITY;
    else value = (1 + mantissa / 1024) * 2 ** (exponent - 15);
    if (sign) value = -value;
    return Math.round(Math.max(0, Math.min(1, Number.isFinite(value) ? value : 1)) * 255);
  }

  private storeArtifact(data: Buffer, mimeType: string): GatewayArtifact {
    this.pruneArtifacts();
    const timestamp = Date.now();
    const artifact = {
      id: randomUUID(),
      mimeType,
      data,
      createdAt: timestamp,
      lastAccessAt: timestamp,
    };
    this.artifacts.set(artifact.id, artifact);
    this.pruneArtifacts();
    return artifact;
  }

  private getArtifact(id: string): GatewayArtifact | undefined {
    if (!/^[0-9a-f-]{36}$/i.test(id)) return undefined;
    this.pruneArtifacts();
    const artifact = this.artifacts.get(id);
    if (artifact) artifact.lastAccessAt = Date.now();
    return artifact;
  }

  private findColorArtifact(value: unknown): GatewayArtifact | undefined {
    if (Array.isArray(value)) {
      for (const entry of value) {
        const found = this.findColorArtifact(entry);
        if (found) return found;
      }
      return undefined;
    }
    if (!value || typeof value !== 'object') return undefined;
    const record = value as Record<string, unknown>;
    if (record.channel === 'color') {
      const artifact = record.artifact;
      if (artifact && typeof artifact === 'object') {
        const id = (artifact as Record<string, unknown>).id;
        if (typeof id === 'string') return this.getArtifact(id);
      }
    }
    for (const nested of Object.values(record)) {
      const found = this.findColorArtifact(nested);
      if (found) return found;
    }
    return undefined;
  }

  private pruneArtifacts(): void {
    const cutoff = Date.now() - artifactLifetimeMilliseconds;
    for (const [id, artifact] of this.artifacts) {
      if (artifact.createdAt < cutoff) this.artifacts.delete(id);
    }
    let total = [...this.artifacts.values()].reduce((sum, artifact) => sum + artifact.data.length, 0);
    if (total <= maximumArtifactBytes) return;
    const oldest = [...this.artifacts.values()].sort((left, right) => left.lastAccessAt - right.lastAccessAt);
    for (const artifact of oldest) {
      this.artifacts.delete(artifact.id);
      total -= artifact.data.length;
      if (total <= maximumArtifactBytes) break;
    }
  }

  private authorized(request: IncomingMessage): boolean {
    const authorization = request.headers.authorization;
    const token = authorization?.startsWith('Bearer ')
      ? authorization.slice(7)
      : typeof request.headers['x-arc-token'] === 'string'
        ? request.headers['x-arc-token']
        : '';
    const actual = Buffer.from(token);
    const expected = Buffer.from(this.token);
    return actual.length === expected.length && timingSafeEqual(actual, expected);
  }

  private validOrigin(request: IncomingMessage): boolean {
    const origin = request.headers.origin;
    if (!origin) return true;
    return origin === this.endpoint || origin.startsWith('arc-editor://');
  }

  private validHost(request: IncomingMessage): boolean {
    if (!this.endpoint) return false;
    const expected = new URL(this.endpoint).host;
    return request.headers.host === expected;
  }

  private clientId(request: IncomingMessage): string {
    const header = request.headers['x-arc-client-id'];
    return typeof header === 'string' && /^[A-Za-z0-9._-]{1,80}$/.test(header) ? header : 'local-client';
  }

  private clientName(request: IncomingMessage): string {
    const header = request.headers['x-arc-client-name'];
    return typeof header === 'string' ? header.slice(0, 120) : 'Local AI client';
  }

  private withinRateLimit(clientId: string): boolean {
    const timestamp = Date.now();
    const current = this.rateLimits.get(clientId);
    if (!current || timestamp - current.started >= 60_000) {
      this.rateLimits.set(clientId, { started: timestamp, count: 1 });
      return true;
    }
    current.count += 1;
    return current.count <= requestsPerMinute;
  }

  private publishEvent(type: string, payload: unknown): void {
    const data = `event: ${type}\ndata: ${JSON.stringify(payload)}\n\n`;
    for (const stream of this.eventStreams) stream.write(data);
  }

  private openApi(): Record<string, unknown> {
    const descriptors = new Map(
      this.harness.operationTransportEntries(this.currentClient()).map(({ descriptor }) => [descriptor.id, descriptor]),
    );
    const wireSchemas = new Map(
      this.harness
        .operationTransportEntries(this.currentClient())
        .map(({ descriptor, input }) => [descriptor.id, z.toJSONSchema(input, { io: 'input' })]),
    );
    const operationPaths = Object.fromEntries(
      Object.entries(this.httpOperations()).map(([apiPath, method]) => [
        apiPath,
        {
          post: {
            operationId: method.replace(/[.-](.)/g, (_match, character: string) => character.toUpperCase()),
            summary: descriptors.get(method)?.description ?? `Invoke ${method}`,
            requestBody: {
              required: false,
              content: {
                'application/json': {
                  schema: wireSchemas.get(method),
                },
              },
            },
            responses: {
              200: {
                description: 'ARC gateway result',
                content: { 'application/json': { schema: descriptors.get(method)?.outputSchema } },
              },
              400: { description: 'Invalid or rejected operation' },
            },
          },
        },
      ]),
    );
    return {
      openapi: '3.1.0',
      info: { title: 'ARC Editor Agent Gateway', version: '1.0.0' },
      servers: [{ url: this.endpoint }],
      security: [{ bearerAuth: [] }],
      paths: {
        '/api/v1/status': {
          get: { operationId: 'gatewayStatus', responses: { 200: { description: 'Gateway status' } } },
        },
        '/api/v1/invoke': {
          post: {
            operationId: 'invokeArcGateway',
            requestBody: {
              required: true,
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    required: ['method'],
                    properties: {
                      method: { type: 'string' },
                      params: { type: 'object', additionalProperties: true },
                    },
                  },
                },
              },
            },
            responses: { 200: { description: 'ARC gateway result' }, 400: { description: 'Invalid request' } },
          },
        },
        ...operationPaths,
      },
      components: {
        securitySchemes: {
          bearerAuth: { type: 'http', scheme: 'bearer', bearerFormat: 'ARC session token' },
        },
      },
      'x-arc-methods': ['gateway.status', ...descriptors.keys()],
    };
  }

  private httpOperations(): Record<string, string> {
    const legacy = new Map<string, string>(Object.entries(gatewayHttpMethods).map(([path, method]) => [method, path]));
    return Object.fromEntries(
      this.harness
        .operationTransportEntries(this.currentClient())
        .map(({ descriptor }) => [legacy.get(descriptor.id) ?? `/api/v1/operations/${descriptor.id}`, descriptor.id]),
    );
  }
}
