import type { ArcProjectCandidate } from '../common/projectTypes';
import { ProjectService as ProjectServiceCore } from './projectServiceCore';
import {
  projectAssetLogicalMounts,
  resolveProjectAssetMountRoots,
  type ProjectAssetLogicalMounts,
} from './projectAssetMountRoots';

type ProjectHost = ConstructorParameters<typeof ProjectServiceCore>[0]['host'];
type ProjectServiceOptions = Omit<ConstructorParameters<typeof ProjectServiceCore>[0], 'host'> & {
  host: ProjectHost;
  userAssetsRoot?: string;
  organizationAssetsRoot?: string;
};

/**
 * Production project-service bridge for host-owned logical asset mounts.
 *
 * The core service remains responsible for project lifecycle/validation. This
 * wrapper owns the storage configuration that crosses the main-process project
 * bridge so renderer code can consume one stable Built-in/Project/User/
 * Organization mount contract without deriving policy from physical paths.
 */
export class ProjectService extends ProjectServiceCore {
  private readonly logicalBuiltinAssetsRoot: string;
  private readonly logicalUserAssetsRoot: string;
  private readonly logicalOrganizationAssetsRoot: string;

  constructor(options: ProjectServiceOptions) {
    const { userAssetsRoot, organizationAssetsRoot, ...coreOptions } = options;
    super(coreOptions);
    this.logicalBuiltinAssetsRoot = options.builtinAssetsRoot ?? '';
    this.logicalUserAssetsRoot = userAssetsRoot ?? process.env.ARC_USER_ASSETS_ROOT ?? '';
    this.logicalOrganizationAssetsRoot =
      organizationAssetsRoot ?? process.env.ARC_ORGANIZATION_ASSETS_ROOT ?? '';
  }

  assetMounts(project: ArcProjectCandidate | null = this.active()): ProjectAssetLogicalMounts {
    if (!project) return {};
    return projectAssetLogicalMounts(
      resolveProjectAssetMountRoots({
        projectRoot: project.projectRoot,
        projectAssetRoots: project.descriptor.assetRoots,
        builtinAssetsRoot: this.logicalBuiltinAssetsRoot,
        userAssetsRoot: this.logicalUserAssetsRoot,
        organizationAssetsRoot: this.logicalOrganizationAssetsRoot,
      }),
    );
  }

  snapshot() {
    return {
      ...super.snapshot(),
      mounts: this.assetMounts(),
    };
  }
}
