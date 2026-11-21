import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { z } from 'zod';
import type { LivePlaySceneBaseline } from './livePlaySceneBaseline';
import { runWorkflowProcess, type WorkflowProcessRequest } from './workflowProcess';

/**
 * L1.3b2c1 prepares a complete *local* target-cooked scene package from the
 * captured unsaved Authoring World. C9 still owns remote release publication;
 * never advertise these files as an active remote generation.
 */
export type LivePlayCookInput = {
  baseline: LivePlaySceneBaseline;
  projectGuid: string;
  projectRoot: string;
  descriptorPath: string;
  assetRoots: readonly string[];
  cooker: string;
  profile: string;
  target: { platform: string; architecture: string; renderer: string };
  signal: AbortSignal;
  isCurrent: () => boolean;
};

export type LivePlayCookedVariant = Readonly<{
  family: string;
  manifestPath: string;
  cookManifestBuildId: string;
  packageDirectory: string;
}>;

export type LivePlayCookedStaging = Readonly<{
  projectGuid: string;
  runtimeBuildId: string;
  profile: string;
  sessionId: string;
  instanceId: string;
  sceneGuid: string;
  sceneAssetGuid: string;
  authoringRevision: number;
  authoringSha256: string;
  directory: string;
  variants: readonly LivePlayCookedVariant[];
}>;

const guidPattern = /^[a-f\d]{8}(?:-[a-f\d]{4}){3}-[a-f\d]{12}$/iu;
const hashPattern = /^[a-f\d]{64}$/u;
const sceneAssetType = 'a7ca55e7-0000-0001-0000-000000000001';
const maxManifestBytes = 16 * 1024 * 1024;
const cookResult = z.object({
  event: z.literal('cook.complete'),
  variants: z.array(z.object({ family: z.string().min(1), manifest: z.string().min(1) })).min(1).max(16),
});
const packageResult = z.object({ event: z.literal('package.complete'), manifest: z.string().min(1) });
const manifestSchema = z.object({
  buildId: z.string().min(1),
  roots: z.array(z.string()).min(1),
  target: z.object({
    name: z.string(),
    platform: z.string(),
    architecture: z.string(),
    renderer: z.string(),
    configuration: z.literal('development'),
  }),
});

function inside(root: string, file: string): boolean {
  const relative = path.relative(root, file);
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function validatedFile(root: string, file: string): string {
  if (!path.isAbsolute(file)) throw new Error('Cooker returned a non-absolute private artifact path');
  const resolved = fs.realpathSync(file);
  if (!inside(root, resolved) || !fs.statSync(resolved).isFile())
    throw new Error('Cooker artifact escaped the session-private staging area');
  return resolved;
}

function parseCompletion<T>(text: string, schema: z.ZodSchema<T>): T {
  const records: T[] = [];
  for (const line of text.split(/\r?\n/u)) {
    if (!line.trim()) continue;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      continue;
    }
    const parsed = schema.safeParse(value);
    if (parsed.success) records.push(parsed.data);
  }
  if (records.length !== 1) throw new Error('Cooker did not provide exactly one valid completion record');
  return records[0];
}

function readManifest(file: string, input: LivePlayCookInput, sceneAssetGuid: string): string {
  const stat = fs.statSync(file);
  if (stat.size === 0 || stat.size > maxManifestBytes) throw new Error('Cook manifest metadata is empty or oversized');
  const manifest = manifestSchema.parse(JSON.parse(fs.readFileSync(file, 'utf8')));
  if (
    manifest.target.name !== input.profile ||
    manifest.target.platform !== input.target.platform ||
    manifest.target.architecture !== input.target.architecture ||
    manifest.target.renderer !== input.target.renderer ||
    !manifest.roots.some((guid) => guid.toLowerCase() === sceneAssetGuid.toLowerCase())
  )
    throw new Error('Cooked scene manifest does not match the target or registered scene asset');
  return manifest.buildId;
}

/** Resolution is from the native autosave's exact source path, not scene.id. */
export function resolveLivePlaySceneAsset(input: LivePlayCookInput): string {
  const projectRoot = fs.realpathSync(input.projectRoot);
  const sceneSource = input.baseline.sourceScenePath;
  if (!sceneSource) throw new Error('Save the scene once before standalone Live Play can cook unsaved changes');
  const scenePath = fs.realpathSync(path.resolve(projectRoot, sceneSource));
  if (!inside(projectRoot, scenePath) || path.extname(scenePath).toLowerCase() !== '.arcscene')
    throw new Error('The active scene source does not belong to the current project');
  const allowed = input.assetRoots.some((root) => {
    const resolved = fs.realpathSync(path.resolve(projectRoot, root));
    return inside(projectRoot, resolved) && inside(resolved, scenePath);
  });
  if (!allowed) throw new Error('The active scene is not in a registered project content root');
  const original = JSON.parse(fs.readFileSync(scenePath, 'utf8')) as { format?: unknown; scene?: { id?: unknown } };
  if (original.format !== 'arc.scene' || original.scene?.id !== input.baseline.sceneGuid)
    throw new Error('The captured scene identity differs from its original saved scene');

  const metadataPath = `${scenePath}.arcmeta`;
  if (!fs.statSync(metadataPath).isFile() || fs.realpathSync(metadataPath) !== metadataPath)
    throw new Error('Scene asset metadata is missing or redirected');
  const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8')) as {
    format?: unknown;
    guid?: unknown;
    type?: unknown;
  };
  if (
    metadata.format !== 'arc.asset-meta' ||
    typeof metadata.guid !== 'string' ||
    !guidPattern.test(metadata.guid) ||
    metadata.type !== sceneAssetType
  )
    throw new Error('The scene is not registered with a valid ARC scene asset GUID');
  return metadata.guid.toLowerCase();
}

export async function prepareLivePlayCookedStaging(
  input: LivePlayCookInput,
  run: (request: WorkflowProcessRequest) => Promise<string> = runWorkflowProcess,
): Promise<LivePlayCookedStaging> {
  const { baseline, signal } = input;
  const assertCurrent = () => {
    signal.throwIfAborted();
    if (!input.isCurrent()) throw new Error('The Live Play project or session changed during scene cooking');
  };
  assertCurrent();
  const projectRoot = fs.realpathSync(input.projectRoot);
  const privateRoot = fs.realpathSync(baseline.directory);
  if (
    baseline.projectGuid !== input.projectGuid ||
    baseline.targetProfile !== input.profile ||
    !baseline.sessionId ||
    !baseline.instanceId ||
    !inside(privateRoot, fs.realpathSync(baseline.scenePath)) ||
    inside(projectRoot, privateRoot)
  )
    throw new Error('Live Play baseline does not belong to this project, profile or private session');

  const source = fs.readFileSync(baseline.scenePath);
  if (source.byteLength !== baseline.byteLength || createHash('sha256').update(source).digest('hex') !== baseline.sha256)
    throw new Error('Live Play scene capture changed after it was sealed');
  if (!hashPattern.test(baseline.sha256)) throw new Error('Invalid captured authoring digest');
  const sceneAssetGuid = resolveLivePlaySceneAsset(input);
  const cooker = fs.realpathSync(input.cooker);
  if (!fs.statSync(cooker).isFile()) throw new Error('ARC cooker tool is unavailable');
  const descriptor = fs.realpathSync(input.descriptorPath);
  if (!inside(projectRoot, descriptor)) throw new Error('Project descriptor is outside the active project');
  const directory = fs.mkdtempSync(path.join(privateRoot, 'cooked-'));
  try {
    assertCurrent();
    const cookDirectory = path.join(directory, 'cook');
    fs.mkdirSync(cookDirectory);
    const cookText = await run({
      executable: cooker,
      args: [
        'cook', '--project', descriptor, '--profile', input.profile,
        '--output', cookDirectory, '--ephemeral-scene', baseline.scenePath,
        '--ephemeral-asset-guid', sceneAssetGuid, '--json',
      ],
      cwd: projectRoot,
      signal,
      timeoutMs: 10 * 60_000,
    });
    assertCurrent();
    const completed = parseCompletion(cookText, cookResult);
    const variants: LivePlayCookedVariant[] = [];
    const seen = new Set<string>();
    for (const variant of completed.variants) {
      assertCurrent();
      const family = variant.family.toLowerCase();
      if (!/^[a-z0-9_-]{1,32}$/u.test(family) || seen.has(family))
        throw new Error('Cooker produced invalid or duplicate texture variants');
      seen.add(family);
      const manifest = validatedFile(cookDirectory, variant.manifest);
      const cookManifestBuildId = readManifest(manifest, input, sceneAssetGuid);
      await run({
        executable: cooker,
        args: ['verify', '--project', descriptor, '--manifest', manifest],
        cwd: projectRoot, signal, timeoutMs: 5 * 60_000,
      });
      assertCurrent();
      const packageDirectory = path.join(directory, 'packages', family);
      fs.mkdirSync(packageDirectory, { recursive: true });
      const packedText = await run({
        executable: cooker,
        args: ['package', '--project', descriptor, '--manifest', manifest, '--output', packageDirectory, '--json'],
        cwd: projectRoot, signal, timeoutMs: 10 * 60_000,
      });
      assertCurrent();
      const packaged = parseCompletion(packedText, packageResult);
      const packagedManifest = validatedFile(packageDirectory, packaged.manifest);
      if (readManifest(packagedManifest, input, sceneAssetGuid) !== cookManifestBuildId)
        throw new Error('Packaged scene manifest has a mismatched cooked build identity');
      await run({
        executable: cooker,
        args: ['verify', '--project', descriptor, '--manifest', packagedManifest],
        cwd: projectRoot, signal, timeoutMs: 5 * 60_000,
      });
      assertCurrent();
      variants.push(Object.freeze({ family, manifestPath: packagedManifest, cookManifestBuildId, packageDirectory }));
    }
    return Object.freeze({
      projectGuid: input.projectGuid,
      runtimeBuildId: baseline.buildId,
      profile: input.profile,
      sessionId: baseline.sessionId,
      instanceId: baseline.instanceId,
      sceneGuid: baseline.sceneGuid,
      sceneAssetGuid,
      authoringRevision: baseline.sceneRevision,
      authoringSha256: baseline.sha256,
      directory,
      variants: Object.freeze(variants),
    });
  } catch (error) {
    fs.rmSync(directory, { recursive: true, force: true });
    throw error;
  }
}
