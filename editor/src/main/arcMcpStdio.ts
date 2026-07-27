import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';

type Discovery = {
  mcpEndpoint: string;
  token: string;
};

const discoveryArgumentIndex = process.argv.indexOf('--discovery');
const discoveryArgument = discoveryArgumentIndex >= 0 ? process.argv[discoveryArgumentIndex + 1] ?? '' : '';
const discoveryCandidates = (): string[] => [
  discoveryArgument,
  process.env.ARC_AI_GATEWAY_DISCOVERY ?? '',
  process.platform === 'win32'
    ? path.join(process.env.APPDATA ?? '', 'arc-editor', 'ai-gateway', 'active.json')
    : path.join(process.env.XDG_CONFIG_HOME ?? path.join(os.homedir(), '.config'),
      'arc-editor', 'ai-gateway', 'active.json'),
].filter(Boolean);

const forward = async (line: string, discovery: Discovery): Promise<void> => {
  if (!line.trim()) return;
  try {
    const parsed = JSON.parse(line) as { method?: string; params?: { name?: string; uri?: string } };
    const headers: Record<string, string> = {
      authorization: `Bearer ${discovery.token}`,
      'content-type': 'application/json',
      accept: 'application/json, text/event-stream',
      'x-arc-client-id': process.env.ARC_AI_CLIENT_ID ?? 'arc-mcp-stdio',
      'x-arc-client-name': process.env.ARC_AI_CLIENT_NAME ?? 'ARC MCP stdio client',
    };
    if (parsed.method) headers['mcp-method'] = parsed.method;
    const name = parsed.params?.name ?? parsed.params?.uri;
    if (name) headers['mcp-name'] = name;
    const response = await fetch(discovery.mcpEndpoint, { method: 'POST', headers, body: line });
    const text = await response.text();
    if (!response.ok) {
      process.stderr.write(`ARC MCP gateway returned HTTP ${response.status}: ${text}\n`);
      return;
    }
    if (response.headers.get('content-type')?.includes('text/event-stream')) {
      for (const eventLine of text.split(/\r?\n/)) {
        if (eventLine.startsWith('data: ')) process.stdout.write(`${eventLine.slice(6)}\n`);
      }
    } else if (text.trim()) {
      process.stdout.write(`${text.trim()}\n`);
    }
  } catch (error) {
    process.stderr.write(`ARC MCP bridge error: ${error instanceof Error ? error.message : String(error)}\n`);
  }
};

const discoveryPath = discoveryCandidates().find((candidate) => fs.existsSync(candidate));
if (!discoveryPath) {
  process.stderr.write('ARC AI gateway discovery file was not found. Start the ARC editor first.\n');
  process.exitCode = 1;
} else {
  const discovery = JSON.parse(fs.readFileSync(discoveryPath, 'utf8')) as Discovery;
  const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  let queue = Promise.resolve();
  input.on('line', (line) => {
    queue = queue.then(() => forward(line, discovery));
  });
}
