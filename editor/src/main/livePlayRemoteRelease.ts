import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { z } from 'zod';
import type { LivePlayCookedStaging } from './livePlayEphemeralCook';

/**
 * L1.3b2c2: C9.1-compatible, immutable *local* publication inventory.
 * The authenticated editor-hosted HTTP server (#1137) remains the only
 * component allowed to advertise these completed generations to clients.
 */
export type LivePlayReleaseInput = {
  staged: LivePlayCookedStaging;
  engineVersion: string;
  generation: number;
  signal: AbortSignal;
  isCurrent: () => boolean;
};

export type LivePlayReleaseVariant = Readonly<{
  family: string;
  descriptorPath: string;
  descriptorSha256: string;
  cookedBuildId: string;
  chunkCount: number;
}>;

export type LivePlayRemoteRelease = Readonly<{
  root: string;
  projectGuid: string;
  runtimeBuildId: string;
  sessionId: string;
  instanceId: string;
  sceneGuid: string;
  sceneAssetGuid: string;
  authoringSha256: string;
  generation: number;
  variants: readonly LivePlayReleaseVariant[];
}>;

const sha256Pattern = /^[a-f0-9]{64}$/u;
const guidPattern = /^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/u;
const familyPattern = /^[a-z0-9_-]{1,32}$/u;
const buildPattern = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u;
const maxManifestBytes = 8 * 1024 * 1024;
const maxChunkBytes = 2 * 1024 * 1024 * 1024;
const maxArtifacts = 100_000;
const maxChunks = 100_000;

const artifactSchema = z.object({
  asset: z.string(),
  hash: z.string().regex(sha256Pattern),
  chunk: z.string(),
  offset: z.number().int().max(Number.MAX_SAFE_INTEGER).nonnegative(),
  size: z.number().int().max(Number.MAX_SAFE_INTEGER).positive(),
  storedSize: z.number().int().max(Number.MAX_SAFE_INTEGER).positive(),
  schemaVersion: z.number().int().positive(),
  compressed: z.literal(false),
});

const manifestSchema = z.object({
  format: z.literal('arc.cook-manifest'),
  version: z.literal(1),
  buildId: z.string().regex(buildPattern),
  roots: z.array(z.string()).min(1),
  target: z.object({
    name: z.string(),
    platform: z.enum(['windows', 'linux', 'macos', 'android']),
    architecture: z.enum(['x86_64', 'arm64']),
    renderer: z.enum(['none', 'vulkan', 'direct3d12', 'metal']),
    textures: z.enum(['bc', 'astc', 'etc2', 'portable']),
    configuration: z.literal('development'),
  }),
  artifacts: z.array(artifactSchema).min(1).max(maxArtifacts),
});

type ObjectReference = Readonly<{ path: string; sha256: string; size: number }>;
type Artifact = z.infer<typeof artifactSchema>;

function within(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function safeFile(root: string, candidate: string): string {
  if (!path.isAbsolute(candidate)) throw new Error('Cooked output must use an absolute private path');
  const stat = fs.lstatSync(candidate);
  if (!stat.isFile() || stat.isSymbolicLink()) throw new Error('Cooked output contains a link or non-file');
  const real = fs.realpathSync(candidate);
  if (!within(root, real)) throw new Error('Cooked output escapes session-private staging');
  return real;
}

function safeName(value: string, extension: string): boolean {
  return value.length <= 255 && /^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(value) && value.endsWith(extension);
}

function lowerCaseSort<T extends { name: string }>(values: T[]): T[] {
  return values.sort((a, b) => a.name.localeCompare(b.name, 'en'));
}

function stableJson(value: unknown): string {
  const normalize = (item: unknown): unknown => {
    if (Array.isArray(item)) return item.map(normalize);
    if (item && typeof item === 'object')
      return Object.fromEntries(
        Object.entries(item)
          .sort(([a], [b]) => a.localeCompare(b, 'en'))
          .map(([key, value]) => [key, normalize(value)]),
      );
    return item;
  };
  return `${JSON.stringify(normalize(value))}\n`;
}

async function digestFile(
  file: string,
  signal: AbortSignal,
  options?: { start: number; end: number },
): Promise<string> {
  const digest = createHash('sha256');
  const source = fs.createReadStream(file, { ...options, highWaterMark: 1024 * 1024, signal });
  for await (const block of source) digest.update(block as Buffer);
  return digest.digest('hex');
}

async function freezeObject(
  source: string,
  objectRoot: string,
  maxBytes: number,
  signal: AbortSignal,
): Promise<{ ref: ObjectReference; file: string }> {
  signal.throwIfAborted();
  const info = fs.statSync(source);
  if (!info.isFile() || info.size < 1 || info.size > maxBytes)
    throw new Error('Cooked source object is empty, invalid, or exceeds publication limits');
  const digest = await digestFile(source, signal);
  const objectFile = path.join(objectRoot, digest);
  try {
    await fs.promises.copyFile(source, objectFile, fs.constants.COPYFILE_EXCL);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
  }
  signal.throwIfAborted();
  if (fs.statSync(objectFile).size !== info.size || (await digestFile(objectFile, signal)) !== digest)
    throw new Error('Cooked content changed while sealing its immutable SHA-256 object');
  return {
    ref: { path: `objects/sha256/${digest}`, sha256: digest, size: info.size },
    file: objectFile,
  };
}

async function verifyChunkArtifacts(
  chunkFile: string,
  name: string,
  artifacts: readonly Artifact[],
  signal: AbortSignal,
): Promise<void> {
  const fileSize = fs.statSync(chunkFile).size;
  const handle = await fs.promises.open(chunkFile, 'r');
  try {
    const header = Buffer.alloc(8);
    const { bytesRead } = await handle.read(header, 0, 8, 0);
    if (bytesRead !== 8 || !header.equals(Buffer.from('ARCPAK1\0', 'utf8')))
      throw new Error(`Cooked chunk ${name} has an invalid package header`);
  } finally {
    await handle.close();
  }
  const ranges: Array<{ begin: number; end: number }> = [];
  for (const artifact of artifacts) {
    signal.throwIfAborted();
    if (
      artifact.offset < 8 ||
      artifact.storedSize !== artifact.size ||
      artifact.offset > fileSize ||
      artifact.storedSize > fileSize - artifact.offset
    )
      throw new Error(`Cooked artifact range exceeds chunk ${name}`);
    const begin = artifact.offset;
    const end = artifact.offset + artifact.storedSize;
    const digest = await digestFile(chunkFile, signal, { start: begin, end: end - 1 });
    if (digest !== artifact.hash) throw new Error(`Cooked artifact hash mismatch in chunk ${name}`);
    ranges.push({ begin, end });
  }
  ranges.sort((a, b) => a.begin - b.begin);
  for (let i = 1; i < ranges.length; i++) {
    if (ranges[i].begin < ranges[i - 1].end) throw new Error(`Cooked artifact ranges overlap in chunk ${name}`);
  }
}

export async function prepareLivePlayRemoteRelease(input: LivePlayReleaseInput): Promise<LivePlayRemoteRelease> {
  const { staged, signal } = input;
  const assertCurrent = () => {
    signal.throwIfAborted();
    if (!input.isCurrent()) throw new Error('Live Play session or project changed while preparing C9 publication');
  };
  assertCurrent();
  if (
    !Number.isSafeInteger(input.generation) ||
    input.generation < 1 ||
    !guidPattern.test(staged.projectGuid) ||
    !guidPattern.test(staged.sceneAssetGuid) ||
    !sha256Pattern.test(staged.authoringSha256) ||
    !staged.sessionId ||
    !staged.instanceId ||
    !staged.runtimeBuildId ||
    !input.engineVersion ||
    !staged.profile ||
    staged.variants.length < 1 ||
    staged.variants.length > 16
  )
    throw new Error('Incomplete or invalid Live Play C9 release identity');

  const stagingRoot = fs.realpathSync(staged.directory);
  if (!fs.statSync(stagingRoot).isDirectory()) throw new Error('Cooked staging directory is invalid');
  const finalRoot = path.join(stagingRoot, 'release');
  if (fs.existsSync(finalRoot)) throw new Error('This Live Play cooked generation has already been published');
  const candidate = fs.mkdtempSync(path.join(stagingRoot, 'release-candidate-'));
  fs.chmodSync(candidate, 0o700);
  const objectRoot = path.join(candidate, 'objects', 'sha256');
  fs.mkdirSync(objectRoot, { recursive: true });
  try {
    const variants: LivePlayReleaseVariant[] = [];
    const seenFamilies = new Set<string>();
    for (const variant of staged.variants) {
      assertCurrent();
      if (!familyPattern.test(variant.family) || seenFamilies.has(variant.family))
        throw new Error('Duplicate or unsafe target texture family');
      seenFamilies.add(variant.family);
      const packageRoot = fs.realpathSync(variant.packageDirectory);
      if (!within(stagingRoot, packageRoot)) throw new Error('Cooked package directory escapes the session');
      const manifestPath = safeFile(packageRoot, variant.manifestPath);
      const manifestName = path.basename(manifestPath);
      if (!safeName(manifestName, '.arccookmanifest') || fs.statSync(manifestPath).size > maxManifestBytes)
        throw new Error('Cooked package manifest has an unsafe name or size');

      const manifestBytes = await fs.promises.readFile(manifestPath, { signal });
      const parsed = manifestSchema.safeParse(JSON.parse(manifestBytes.toString('utf8')));
      if (!parsed.success) throw new Error('Cooked package manifest does not meet ARC v1 requirements');
      const manifest = parsed.data;
      if (
        manifest.buildId !== variant.cookManifestBuildId ||
        manifest.target.name !== staged.profile ||
        manifest.target.textures !== variant.family ||
        !manifest.roots.some((root) => root.toLowerCase() === staged.sceneAssetGuid)
      )
        throw new Error('Cooked package build, profile, family or source asset identity differs');
      if (!manifest.artifacts.some((artifact) => artifact.asset.toLowerCase() === staged.sceneAssetGuid))
        throw new Error('Cooked package does not contain the required scene artifact');

      const grouped = new Map<string, Artifact[]>();
      const foldedNames = new Set<string>();
      for (const artifact of manifest.artifacts) {
        if (!safeName(artifact.chunk, '.arcpak')) throw new Error('Cooked package manifest declares an unsafe chunk');
        if (!grouped.has(artifact.chunk)) grouped.set(artifact.chunk, []);
        grouped.get(artifact.chunk)!.push(artifact);
      }
      if (grouped.size > maxChunks) throw new Error('Too many cooked chunks in release');
      for (const name of grouped.keys()) {
        const folded = name.toLowerCase();
        if (foldedNames.has(folded)) throw new Error('Case-insensitive cooked chunk collision');
        foldedNames.add(folded);
      }
      const actualChunks = fs.readdirSync(packageRoot).filter((name) => name.toLowerCase().endsWith('.arcpak'));
      if (actualChunks.length !== grouped.size || actualChunks.some((name) => !grouped.has(name)))
        throw new Error('Cooked package file set differs from the complete manifest closure');

      const frozenManifest = await freezeObject(manifestPath, objectRoot, maxManifestBytes, signal);
      const chunks: Array<{ name: string } & ObjectReference> = [];
      const hashes = new Set<string>([frozenManifest.ref.sha256]);
      for (const [name, artifacts] of grouped) {
        assertCurrent();
        const source = safeFile(packageRoot, path.join(packageRoot, name));
        const frozen = await freezeObject(source, objectRoot, maxChunkBytes, signal);
        if (hashes.has(frozen.ref.sha256)) throw new Error('Duplicate immutable object in a cooked release');
        hashes.add(frozen.ref.sha256);
        await verifyChunkArtifacts(frozen.file, name, artifacts, signal);
        chunks.push({ name, ...frozen.ref });
      }
      assertCurrent();
      const release = {
        format: 'arc.remote-content',
        formatVersion: 1,
        projectGuid: staged.projectGuid,
        engineVersion: input.engineVersion,
        baseBuildId: staged.runtimeBuildId,
        cookBuildId: manifest.buildId,
        target: {
          platform: manifest.target.platform,
          architecture: manifest.target.architecture,
          renderer: manifest.target.renderer,
          cookProfile: staged.profile,
          textureFamily: variant.family,
        },
        generation: input.generation,
        complete: true,
        cookManifestName: manifestName,
        cookManifest: frozenManifest.ref,
        chunks: lowerCaseSort(chunks),
      };
      const metadata = stableJson(release);
      const targetFolder = path.join(candidate, 'targets', variant.family);
      fs.mkdirSync(targetFolder, { recursive: true });
      const descriptorPath = path.join(targetFolder, 'current.json');
      fs.writeFileSync(descriptorPath, metadata, { encoding: 'utf8', flag: 'wx', mode: 0o600 });
      variants.push({
        family: variant.family,
        descriptorPath: path.join(finalRoot, 'targets', variant.family, 'current.json'),
        descriptorSha256: createHash('sha256').update(metadata).digest('hex'),
        cookedBuildId: manifest.buildId,
        chunkCount: chunks.length,
      });
    }
    assertCurrent();
    if (fs.existsSync(finalRoot)) throw new Error('Concurrent Live Play publication already exists');
    await fs.promises.rename(candidate, finalRoot);
    return Object.freeze({
      root: finalRoot,
      projectGuid: staged.projectGuid,
      runtimeBuildId: staged.runtimeBuildId,
      sessionId: staged.sessionId,
      instanceId: staged.instanceId,
      sceneGuid: staged.sceneGuid,
      sceneAssetGuid: staged.sceneAssetGuid,
      authoringSha256: staged.authoringSha256,
      generation: input.generation,
      variants: Object.freeze(variants.map((variant) => Object.freeze(variant))),
    });
  } catch (error) {
    fs.rmSync(candidate, { recursive: true, force: true });
    throw error;
  }
}
