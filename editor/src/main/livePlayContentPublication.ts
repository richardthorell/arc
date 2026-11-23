import type { LivePlaySceneBaseline } from './livePlaySceneBaseline';
import { prepareLivePlayCookedStaging, type LivePlayCookInput } from './livePlayEphemeralCook';
import { prepareLivePlayRemoteRelease, type LivePlayRemoteRelease } from './livePlayRemoteRelease';
import type { WorkflowProcessRequest } from './workflowProcess';
import { EditorCookedContentServer } from './editorCookedContentServer';

/**
 * C9.4b: trusted editor-main orchestration for the initial standalone Play
 * generation. The capture is already sealed by L1.3a. The cooker verifies a
 * private package, C9.1 freezes it, and only then C9.4a promotes it over HTTP.
 *
 * C9.4c will securely hand the authenticated endpoint to native clients.
 * No credentials or authoring paths are returned from this coordinator.
 */
export type LivePlayContentPublicationInput = Readonly<{
  baseline: LivePlaySceneBaseline;
  projectGuid: string;
  projectRoot: string;
  descriptorPath: string;
  assetRoots: readonly string[];
  cooker: string;
  profile: string;
  target: LivePlayCookInput['target'];
  textureFamilies: readonly string[];
  engineVersion: string;
  server: EditorCookedContentServer;
  isCurrent: () => boolean;
}>;

export type LivePlayPublishedContent = Readonly<{
  projectGuid: string;
  profile: string;
  textureFamily: string;
  generation: number;
  sceneGuid: string;
  sceneAssetGuid: string;
  descriptorSha256: string;
  objectCount: number;
}>;

type CookRunner = (request: WorkflowProcessRequest) => Promise<string>;

/**
 * A cook profile can advertise several possible texture families. Do not
 * silently publish the first one and misrepresent a target as compatible.
 * Target/device-specific selection for multiple variants arrives in C9.4c.
 */
export function selectLivePlayTextureFamily(
  release: LivePlayRemoteRelease,
  configuredFamilies: readonly string[],
): string {
  const familyPattern = /^[a-z0-9_-]{1,32}$/u;
  const allowed = new Set(configuredFamilies);
  if (!allowed.size || [...allowed].some((family) => !familyPattern.test(family)))
    throw new Error('The selected cook profile has an invalid texture-family policy');
  const families = release.variants.map((variant) => variant.family);
  if (new Set(families).size !== families.length || families.some((family) => !familyPattern.test(family)))
    throw new Error('The cooked release contains unsafe or duplicate target variants');
  const compatible = families.filter((family) => allowed.has(family));
  if (compatible.length !== 1)
    throw new Error(
      compatible.length === 0
        ? 'No cooked texture family matches the selected target profile'
        : 'Multiple compatible cooked variants require an explicit target texture-family selection',
    );
  return compatible[0];
}

export class LivePlayContentPublication {
  private readonly controller = new AbortController();
  private pending: Promise<LivePlayPublishedContent> | null = null;

  constructor(
    private readonly input: LivePlayContentPublicationInput,
    private readonly runCook?: CookRunner,
  ) {}

  /** One initial release per captured session; later live revisions are C9.5. */
  start(): Promise<LivePlayPublishedContent> {
    if (this.pending || this.controller.signal.aborted)
      throw new Error('C9 initial publication already started or cancelled');
    this.pending = this.publish();
    return this.pending;
  }

  /** Called BEFORE private staging cleanup on stop, project change or quit. */
  async cancel(): Promise<void> {
    this.controller.abort();
    try {
      await this.pending;
    } catch {
      // The pending task owns its failure; cancellation must always finish.
    }
  }

  private assertCurrent(): void {
    this.controller.signal.throwIfAborted();
    if (!this.input.isCurrent()) throw new Error('C9 publication session or active project changed');
  }

  private async publish(): Promise<LivePlayPublishedContent> {
    const input = this.input;
    this.assertCurrent();
    if (
      input.baseline.projectGuid !== input.projectGuid ||
      input.baseline.targetProfile !== input.profile ||
      !input.baseline.sessionId ||
      !input.baseline.instanceId ||
      !input.engineVersion ||
      !input.cooker ||
      !input.textureFamilies.length
    )
      throw new Error('C9 publication requires complete, matching trusted session and target identities');

    const cooked = await prepareLivePlayCookedStaging(
      {
        baseline: input.baseline,
        projectGuid: input.projectGuid,
        projectRoot: input.projectRoot,
        descriptorPath: input.descriptorPath,
        assetRoots: input.assetRoots,
        cooker: input.cooker,
        profile: input.profile,
        target: input.target,
        signal: this.controller.signal,
        isCurrent: () => !this.controller.signal.aborted && input.isCurrent(),
      },
      this.runCook,
    );
    this.assertCurrent();
    if (
      cooked.runtimeBuildId !== input.baseline.buildId ||
      cooked.sessionId !== input.baseline.sessionId ||
      cooked.instanceId !== input.baseline.instanceId
    )
      throw new Error('The cooked C9 generation does not belong to this Live Play session');

    const release = await prepareLivePlayRemoteRelease({
      staged: cooked,
      engineVersion: input.engineVersion,
      generation: 1,
      signal: this.controller.signal,
      isCurrent: () => !this.controller.signal.aborted && input.isCurrent(),
    });
    this.assertCurrent();
    const family = selectLivePlayTextureFamily(release, input.textureFamilies);
    const variant = release.variants.find((value) => value.family === family);
    if (!variant) throw new Error('The compatible cooked release descriptor is missing');

    // Do not promote before the final session check; C9.4a independently
    // verifies the sealed descriptor and all manifest/chunk SHA-256 objects.
    await input.server.publish(release, input.profile, family);
    this.assertCurrent();
    return Object.freeze({
      projectGuid: release.projectGuid,
      profile: input.profile,
      textureFamily: family,
      generation: release.generation,
      sceneGuid: release.sceneGuid,
      sceneAssetGuid: release.sceneAssetGuid,
      descriptorSha256: variant.descriptorSha256,
      objectCount: variant.chunkCount + 1,
    });
  }
}
