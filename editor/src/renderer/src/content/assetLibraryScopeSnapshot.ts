import type { AssetItem, ProjectSnapshot } from '../services/editorHostTypes';
import {
  buildAssetLibraryMountNavigation,
  buildAssetLibraryMounts,
  type AssetLibraryMountMap,
} from './assetLibraryMounts';
import { buildAssetLibraryScopeViews, type AssetLibraryScopeView } from './assetLibraryScopeView';

/**
 * Host-owned storage configuration supplied with an editor project snapshot.
 * Physical roots are consumed only to establish logical mount availability and
 * never become part of renderer navigation state or asset identity.
 */
export type AssetLibraryHostScopeConfig = {
  mounts?: AssetLibraryMountMap;
};

export type ProjectSnapshotWithAssetLibraryScopes = ProjectSnapshot & AssetLibraryHostScopeConfig;

/**
 * Builds the Content Browser's logical scope model from authoritative host
 * configuration plus the authoritative asset registry snapshot.
 *
 * Project keeps the existing `assetRoot` as a compatibility fallback while the
 * optional Built-in/User/Organization mounts must be explicitly configured by
 * the host. This avoids inferring mount availability from whether a scope
 * happens to contain assets.
 */
export function assetLibraryScopeViewsForProject(
  project: ProjectSnapshotWithAssetLibraryScopes | null,
  assets: readonly AssetItem[] = project?.assets ?? [],
): AssetLibraryScopeView[] {
  const roots: AssetLibraryMountMap = project
    ? {
        ...project.mounts,
        project: project.mounts?.project ?? project.assetRoot,
      }
    : {};

  return buildAssetLibraryScopeViews(buildAssetLibraryMountNavigation(buildAssetLibraryMounts(roots)), assets);
}
