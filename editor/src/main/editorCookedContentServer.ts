import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';
import fs from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { Socket } from 'node:net';
import path from 'node:path';
import type { LivePlayRemoteRelease } from './livePlayRemoteRelease';

/** C9.4a: main-process-owned read-only serving of already verified L1.3 output.
 * No renderer API, source-tree route, project SQL query, or untrusted filename.
 */
export type EditorContentServerOptions = Readonly<{
  projectRoot: string;
  sessionRoot: string;
  projectGuid: string;
  runtimeBuildId: string;
  sessionId: string;
  instanceId: string;
  port?: number;
}>;

export type EditorContentServerSnapshot = Readonly<{
  state: 'stopped' | 'listening';
  origin: string | null;
  projectGuid: string;
  sessionId: string;
  profiles: ReadonlyArray<{ profile: string; family: string; generation: number }>;
  activeClients: number;
  activeTransfers: number;
  requests: number;
  bytesSent: number;
  rejectedRequests: number;
}>;

type ObjectRecord = Readonly<{
  file: string;
  size: number;
  sha256: string;
  dev: number;
  ino: number;
  mtimeMs: number;
  ctimeMs: number;
}>;

type Publication = Readonly<{ descriptor: Buffer; hash: string; generation: number; family: string }>;

const guidPattern = /^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/u;
const digestPattern = /^[a-f0-9]{64}$/u;
const profilePattern = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u;
const familyPattern = /^[a-z0-9_-]{1,32}$/u;
const releasePath = /^\/releases\/([a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})\/([A-Za-z0-9][A-Za-z0-9._-]{0,127})\/current\.json$/u;
const objectPath = /^\/objects\/sha256\/([a-f0-9]{64})$/u;
const maxDescriptor = 8 * 1024 * 1024;
const maxTransfer = 256 * 1024 * 1024;
const noStore = 'no-store, no-cache, must-revalidate';

function within(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative !== '' && relative !== '..' && !relative.startsWith('..' + path.sep) && !path.isAbsolute(relative);
}

function sealedFile(root: string, candidate: string): string {
  if (!path.isAbsolute(candidate) || !within(root, candidate)) throw new Error('Unapproved private cooked path');
  const stat = fs.lstatSync(candidate);
  if (!stat.isFile() || stat.isSymbolicLink() || fs.realpathSync(candidate) !== candidate)
    throw new Error('Private cooked object has a symlink or invalid file');
  return candidate;
}

async function hashFile(file: string): Promise<string> {
  const digest = createHash('sha256');
  for await (const block of fs.createReadStream(file, { highWaterMark: 1024 * 1024 }))
    digest.update(block as Buffer);
  return digest.digest('hex');
}

function parseRelease(bytes: Buffer, release: LivePlayRemoteRelease, cookProfile: string, family: string) {
  if (!bytes.length || bytes.length > maxDescriptor) throw new Error('Invalid C9 descriptor size');
  const raw: unknown = JSON.parse(bytes.toString('utf8'));
  if (!raw || typeof raw !== 'object') throw new Error('Invalid C9 release descriptor');
  const data = raw as Record<string, unknown>;
  const target = data.target as Record<string, unknown> | undefined;
  if (
    data.format !== 'arc.remote-content' ||
    data.formatVersion !== 1 ||
    data.complete !== true ||
    data.projectGuid !== release.projectGuid ||
    data.baseBuildId !== release.runtimeBuildId ||
    data.generation !== release.generation ||
    !target ||
    target.cookProfile !== cookProfile ||
    target.textureFamily !== family ||
    !Array.isArray(data.chunks) ||
    data.chunks.length > 100_000
  ) throw new Error('C9 descriptor does not match the approved runtime, target or generation');
  const entries = [data.cookManifest, ...data.chunks];
  const objects: Array<{ sha256: string; size: number }> = [];
  const seen = new Set<string>();
  for (const entry of entries) {
    if (!entry || typeof entry !== 'object') throw new Error('Invalid C9 object reference');
    const ref = entry as Record<string, unknown>;
    if (
      typeof ref.sha256 !== 'string' ||
      !digestPattern.test(ref.sha256) ||
      ref.path !== 'objects/sha256/' + ref.sha256 ||
      typeof ref.size !== 'number' ||
      !Number.isSafeInteger(ref.size) ||
      ref.size < 1 ||
      ref.size > 2 * 1024 * 1024 * 1024 ||
      seen.has(ref.sha256)
    ) throw new Error('Unsafe or duplicate immutable object reference');
    seen.add(ref.sha256);
    objects.push({ sha256: ref.sha256, size: ref.size });
  }
  return objects;
}

/** Endpoint credentials are only available to trusted editor-main callers.
 * Future C9.4b passes them over the existing protected native handoff, never
 * through URL, CLI, project settings, log, or renderer IPC.
 */
export class EditorCookedContentServer {
  private server: Server | null = null;
  private readonly sockets = new Set<Socket>();
  private readonly publications = new Map<string, Publication>();
  private readonly objects = new Map<string, ObjectRecord>();
  private readonly projectRoot: string;
  private readonly sessionRoot: string;
  private port: number | null = null;
  private authorizationHash: Buffer | null = null;
  private authorizationExpires = 0;
  private requests = 0;
  private bytesSent = 0;
  private rejectedRequests = 0;
  private activeTransfers = 0;
  private windowStart = 0;
  private windowRequests = 0;
  private publishing = false;

  constructor(private readonly options: EditorContentServerOptions) {
    if (
      !guidPattern.test(options.projectGuid) ||
      !options.runtimeBuildId || !options.sessionId || !options.instanceId ||
      (options.port !== undefined &&
        (!Number.isInteger(options.port) || options.port < 0 || options.port > 65535))
    ) throw new Error('C9 editor content server requires a trusted session identity');
    this.projectRoot = fs.realpathSync(options.projectRoot);
    this.sessionRoot = fs.realpathSync(options.sessionRoot);
    if (
      !fs.statSync(this.projectRoot).isDirectory() ||
      !fs.statSync(this.sessionRoot).isDirectory() ||
      !path.isAbsolute(options.sessionRoot) ||
      this.projectRoot === this.sessionRoot ||
      within(this.projectRoot, this.sessionRoot)
    ) throw new Error('C9 content session must be outside the authoring workspace');
  }

  snapshot(): EditorContentServerSnapshot {
    return {
      state: this.server ? 'listening' : 'stopped',
      origin: this.port === null ? null : 'http://127.0.0.1:' + this.port,
      projectGuid: this.options.projectGuid,
      sessionId: this.options.sessionId,
      profiles: [...this.publications].map(([profile, p]) => ({
        profile, family: p.family, generation: p.generation,
      })),
      activeClients: this.sockets.size,
      activeTransfers: this.activeTransfers,
      requests: this.requests,
      bytesSent: this.bytesSent,
      rejectedRequests: this.rejectedRequests,
    };
  }

  async start(): Promise<void> {
    if (this.server) throw new Error('C9 content listener is already active');
    const server = createServer((req, res) => this.handle(req, res));
    server.maxConnections = 16;
    server.headersTimeout = 5000;
    server.requestTimeout = 8000;
    server.keepAliveTimeout = 2000;
    server.on('connection', (socket) => {
      this.sockets.add(socket);
      socket.on('close', () => this.sockets.delete(socket));
    });
    try {
      await new Promise<void>((resolve, reject) => {
        server.once('error', reject);
        server.listen(this.options.port ?? 0, '127.0.0.1', () => {
          server.off('error', reject);
          resolve();
        });
      });
      const address = server.address();
      if (!address || typeof address === 'string') throw new Error('C9 listener has no local port');
      this.server = server;
      this.port = address.port;
      server.on('error', () => {
        this.publications.clear();
        this.objects.clear();
        for (const socket of this.sockets) socket.destroy();
      });
    } catch (error) {
      try { server.close(); } catch { /* failed listen has no open port */ }
      throw new Error('Unable to bind C9 loopback listener: ' +
        (error instanceof Error ? error.message : 'unknown error'));
    }
  }

  issueClientToken(): string {
    if (!this.server) throw new Error('C9 listener is not active');
    const token = randomBytes(32).toString('hex');
    this.authorizationHash = createHash('sha256').update(token).digest();
    this.authorizationExpires = Date.now() + 15 * 60_000;
    return token;
  }

  /** Verify a complete release before atomically advertising its descriptor
   * and object inventory. Failed candidates leave the old generation visible.
   */
  async publish(release: LivePlayRemoteRelease, cookProfile: string, family?: string): Promise<void> {
    const originalServer = this.server;
    if (!originalServer) throw new Error('C9 listener is not active');
    if (this.publishing) throw new Error('C9 publication is already being verified');
    this.publishing = true;
    try {
    if (
      release.projectGuid !== this.options.projectGuid ||
      release.runtimeBuildId !== this.options.runtimeBuildId ||
      release.sessionId !== this.options.sessionId ||
      release.instanceId !== this.options.instanceId ||
      !Number.isSafeInteger(release.generation) || release.generation < 1 ||
      !profilePattern.test(cookProfile)
    ) throw new Error('C9 release belongs to a different project, target or session');
    if (!release.variants.length || (release.variants.length > 1 && !family))
      throw new Error('Explicit texture family required for multi-variant C9 publication');

    const root = fs.realpathSync(release.root);
    if (
      !within(this.sessionRoot, root) ||
      path.basename(root) !== 'release' ||
      fs.lstatSync(release.root).isSymbolicLink() ||
      !fs.statSync(root).isDirectory()
    ) throw new Error('Release is outside sealed private session storage');
    const variant = family ? release.variants.find((item) => item.family === family) : release.variants[0];
    if (!variant || !familyPattern.test(variant.family)) throw new Error('Unknown C9 texture family');
    const current = this.publications.get(cookProfile);
    if (current && release.generation <= current.generation)
      throw new Error('C9 generation must increase monotonically for the selected target');

    const descriptorFile = sealedFile(root, variant.descriptorPath);
    if (descriptorFile !== path.join(root, 'targets', variant.family, 'current.json'))
      throw new Error('C9 descriptor is outside its selected target inventory');
    const descriptor = await fs.promises.readFile(descriptorFile);
    const hash = createHash('sha256').update(descriptor).digest('hex');
    if (hash !== variant.descriptorSha256) throw new Error('C9 descriptor SHA-256 changed');
    const references = parseRelease(descriptor, release, cookProfile, variant.family);

    const nextObjects = new Map(this.objects);
    for (const reference of references) {
      const file = sealedFile(root, path.join(root, 'objects', 'sha256', reference.sha256));
      const stat = fs.statSync(file);
      if (stat.size !== reference.size || (await hashFile(file)) !== reference.sha256)
        throw new Error('C9 object failed size/hash verification');
      const old = nextObjects.get(reference.sha256);
      if (old && old.size !== reference.size) throw new Error('Incompatible C9 object identity');
      nextObjects.set(reference.sha256, {
        file, size: reference.size, sha256: reference.sha256,
        dev: stat.dev, ino: stat.ino, mtimeMs: stat.mtimeMs, ctimeMs: stat.ctimeMs,
      });
    }

    if (this.server !== originalServer) throw new Error('C9 listener changed during release verification');
    const nextPublications = new Map(this.publications);
    nextPublications.set(cookProfile, { descriptor, hash, generation: release.generation, family: variant.family });
    this.objects.clear();
    for (const [name, entry] of nextObjects) this.objects.set(name, entry);
    this.publications.clear();
    for (const [name, entry] of nextPublications) this.publications.set(name, entry);
    } finally {
      this.publishing = false;
    }
  }

  async stop(): Promise<void> {
    const server = this.server;
    this.server = null;
    this.port = null;
    this.publications.clear();
    this.objects.clear();
    this.authorizationHash?.fill(0);
    this.authorizationHash = null;
    this.authorizationExpires = 0;
    for (const socket of this.sockets) socket.destroy();
    this.sockets.clear();
    if (!server) return;
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }

  private authorized(req: IncomingMessage): boolean {
    const header = req.headers.authorization;
    if (!this.authorizationHash || Date.now() >= this.authorizationExpires || typeof header !== 'string')
      return false;
    const match = /^Bearer ([a-f0-9]{64})$/u.exec(header);
    if (!match) return false;
    return timingSafeEqual(createHash('sha256').update(match[1]).digest(), this.authorizationHash);
  }

  private reject(res: ServerResponse, code: number, message: string): void {
    this.rejectedRequests++;
    res.writeHead(code, {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': noStore,
      'X-Content-Type-Options': 'nosniff',
      ...(code === 401 ? { 'WWW-Authenticate': 'Bearer' } : {}),
    });
    res.end(message);
  }

  private handle(req: IncomingMessage, res: ServerResponse): void {
    this.requests++;
    const now = Date.now();
    if (now - this.windowStart >= 1000) {
      this.windowStart = now;
      this.windowRequests = 0;
    }
    if (++this.windowRequests > 100) return this.reject(res, 429, 'Request limit exceeded');
    if (!this.server || this.port === null) return this.reject(res, 503, 'Content server stopping');
    if (req.socket.remoteAddress !== '127.0.0.1') return this.reject(res, 403, 'Loopback only');
    if (req.headers.host !== '127.0.0.1:' + this.port)
      return this.reject(res, 400, 'Invalid content authority');
    if (req.rawHeaders.length / 2 > 64 || req.headers['content-length'] !== undefined)
      return this.reject(res, 431, 'Invalid request headers');
    if (!this.authorized(req)) return this.reject(res, 401, 'Content pairing required');
    if (req.method !== 'GET') return this.reject(res, 405, 'Only read-only GET is supported');
    const requested = req.url ?? '';
    if (
      requested.length > 512 || requested.includes('%') ||
      requested.includes('?') || requested.includes('#') || requested.includes('\\')
    ) return this.reject(res, 400, 'Unsafe content URL');

    const release = releasePath.exec(requested);
    if (release) {
      if (release[1] !== this.options.projectGuid) return this.reject(res, 404, 'Unknown project');
      const entry = this.publications.get(release[2]);
      if (!entry) return this.reject(res, 404, 'No verified release for target');
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Content-Length': entry.descriptor.length,
        'Cache-Control': noStore,
        'ETag': '"' + entry.hash + '"',
        'X-Content-Type-Options': 'nosniff',
      });
      res.end(entry.descriptor);
      this.bytesSent += entry.descriptor.length;
      return;
    }

    const object = objectPath.exec(requested);
    if (!object) return this.reject(res, 404, 'Unknown cooked endpoint');
    const entry = this.objects.get(object[1]);
    if (!entry) return this.reject(res, 404, 'Unpublished object');
    const etag = '"' + entry.sha256 + '"';
    if (req.headers['if-match'] && req.headers['if-match'] !== etag)
      return this.reject(res, 412, 'ETag mismatch');
    let start = 0;
    let end = entry.size - 1;
    const range = req.headers.range;
    if (range !== undefined) {
      if (typeof range !== 'string') return this.reject(res, 400, 'Invalid range header');
      const match = /^bytes=(0|[1-9][0-9]*)-(0|[1-9][0-9]*)$/u.exec(range);
      if (!match) return this.reject(res, 400, 'Only explicit bounded byte ranges are supported');
      start = Number(match[1]);
      end = Number(match[2]);
      if (
        !Number.isSafeInteger(start) || !Number.isSafeInteger(end) ||
        start > end || end >= entry.size || end - start + 1 > maxTransfer
      ) {
        res.setHeader('Content-Range', 'bytes */' + entry.size);
        return this.reject(res, 416, 'Range not satisfiable');
      }
    } else if (entry.size > maxTransfer) {
      return this.reject(res, 413, 'Large cooked objects require bounded ranges');
    }

    try {
      const stat = fs.lstatSync(entry.file);
      if (
        !stat.isFile() || stat.isSymbolicLink() ||
        stat.size !== entry.size || stat.dev !== entry.dev || stat.ino !== entry.ino ||
        stat.mtimeMs !== entry.mtimeMs || stat.ctimeMs !== entry.ctimeMs ||
        fs.realpathSync(entry.file) !== entry.file
      ) return this.reject(res, 503, 'Published object changed after verification');
      const stream = fs.createReadStream(entry.file, { start, end, highWaterMark: 64 * 1024 });
      this.activeTransfers++;
      res.once('close', () => {
        this.activeTransfers = Math.max(0, this.activeTransfers - 1);
        stream.destroy();
      });
      stream.once('error', () => {
        if (!res.headersSent) this.reject(res, 503, 'Cooked object unavailable');
        else res.destroy();
      });
      res.once('finish', () => { this.bytesSent += end - start + 1; });
      const headers: Record<string, string | number> = {
        'Content-Type': 'application/octet-stream',
        'Content-Length': end - start + 1,
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'private, max-age=31536000, immutable',
        'ETag': etag,
        'X-Content-Type-Options': 'nosniff',
      };
      if (range !== undefined) headers['Content-Range'] = 'bytes ' + start + '-' + end + '/' + entry.size;
      res.writeHead(range !== undefined ? 206 : 200, headers);
      stream.pipe(res);
    } catch {
      this.reject(res, 503, 'Cooked object unavailable');
    }
  }
}
