import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import type { ArcProjectCandidate } from '../common/projectTypes';
import type { WorkflowRequest, WorkflowRun, WorkflowExecutionSnapshot } from '../common/projectWorkflowTypes';
import { workflowRequestSchema } from '../common/projectWorkflowTypes';
import { queryProjectWorkflowPlan } from './projectWorkflowPlanner';
import { createCookWorkflowAdapter } from './cookWorkflowAdapter';
import { ProjectWorkflowRunner } from './projectWorkflowRunner';
import type { CookWorkflowOptions } from '../common/projectWorkflowTypes';
import { createWindowsBuildWorkflow, type WindowsBuildEnvironment } from './windowsBuildWorkflow';

/** Resolve only known locations belonging to the selected project-tool installation/build. */
export function resolveWorkflowCooker(projectTool: string): string {
  if (!projectTool) return '';
  const executable = process.platform === 'win32' ? 'arc-cook.exe' : 'arc-cook';
  const directory = path.dirname(projectTool);
  const candidates = [path.join(directory, executable)];
  if (path.basename(directory) === 'project_cli')
    candidates.push(path.join(directory, '..', 'asset_cooker', executable));
  else if (path.basename(path.dirname(directory)) === 'project_cli')
    candidates.push(path.join(directory, '..', '..', 'asset_cooker', path.basename(directory), executable));
  return (
    candidates.find((candidate) => {
      try {
        return fs.statSync(candidate).isFile();
      } catch {
        return false;
      }
    }) ?? ''
  );
}

export class CookWorkflowService {
  private runner?: ProjectWorkflowRunner;
  private busy = false;
  private cancelled = false;
  private authorityEpoch = 0;
  private value: WorkflowExecutionSnapshot = { requestId: null, planning: false, run: null, output: [], error: null };

  constructor(
    private readonly project: () => ArcProjectCandidate | null,
    private readonly tool: () => string,
    private readonly legacyBusy: () => boolean,
    private readonly publish: (snapshot: WorkflowExecutionSnapshot) => void,
    private readonly cooker: (tool: string) => string = resolveWorkflowCooker,
    private readonly buildEnvironment: () => WindowsBuildEnvironment | undefined = () => undefined,
  ) {}

  isBusy(): boolean {
    return this.busy;
  }
  snapshot(): WorkflowExecutionSnapshot {
    return structuredClone(this.value);
  }
  private emit(): void {
    try {
      this.publish(this.snapshot());
    } catch {
      /* Observer only. */
    }
  }
  cancel(requestId: string): void {
    if (!this.busy || this.value.requestId !== requestId) return;
    this.cancelled = true;
    const run = this.runner?.snapshot();
    if (run) this.runner?.cancel(run.id);
  }
  invalidate(): void {
    ++this.authorityEpoch;
    if (this.value.requestId) this.cancel(this.value.requestId);
  }

  private selectionAuthority(): string {
    const project = this.project();
    if (!project) throw new Error('No project is open');
    const executableIdentity = (executable: string) => {
      try {
        const stat = fs.statSync(executable);
        return { path: fs.realpathSync(executable), size: stat.size, modified: stat.mtimeMs };
      } catch {
        return { path: executable, unavailable: true };
      }
    };
    return createHash('sha256')
      .update(
        JSON.stringify({
          path: project.descriptorPath,
          root: project.projectRoot,
          descriptor: project.descriptor,
          saved: fs.readFileSync(project.descriptorPath, 'utf8'),
          epoch: this.authorityEpoch,
          tool: executableIdentity(this.tool()),
          cooker: executableIdentity(this.cooker(this.tool())),
          buildEnvironment: this.buildEnvironment(),
        }),
      )
      .digest('hex');
  }

  options(): CookWorkflowOptions {
    const project = this.project();
    if (!project) return { authority: '', profiles: [], unavailableReason: 'Open a project to cook content' };
    const authority = this.selectionAuthority();
    const profiles = project.descriptor.cookProfiles.flatMap((profile) => {
      // This is a locator for an authored target, not an executable stage plan.
      // The native planner remains authoritative and validates the pair before execution.
      const targetId = `${profile.platform}-${profile.architecture === 'x86_64' ? 'x64' : profile.architecture}-${profile.renderer === 'none' ? 'headless' : profile.renderer}`;
      if (!project.descriptor.targetPlatforms.some((target) => target.enabled && target.id === targetId)) return [];
      return [
        {
          id: profile.id,
          targetId,
          platform: profile.platform,
          label: `${profile.id} — ${profile.configuration} / ${profile.architecture} / ${profile.renderer} / ${profile.textures.outputs.join(', ')}`,
        },
      ];
    });
    return {
      authority,
      profiles,
      unavailableReason: !project.writable
        ? 'Project is read-only'
        : !this.cooker(this.tool())
          ? 'ARC cooker is unavailable in the selected SDK'
          : undefined,
    };
  }

  async execute(raw: WorkflowRequest, expectedAuthority?: string): Promise<WorkflowRun> {
    if (
      expectedAuthority !== undefined &&
      (typeof expectedAuthority !== 'string' || expectedAuthority !== this.selectionAuthority())
    )
      throw new Error('Project or SDK changed; reopen Cook Content Only before continuing');
    const request = workflowRequestSchema.parse(raw);
    if (!['cook', 'build'].includes(request.action) || request.destination !== 'none' || request.sessionId)
      throw new Error('Only saved-content Cook and Build workflows are executable; launch adapters are not ready');
    if (this.busy || this.legacyBusy()) throw new Error('A project workflow or code build is already running');
    const project = this.project();
    if (!project?.writable) throw new Error('Cooking requires a writable open project');
    const tool = this.tool();
    const cooker = this.cooker(tool);
    const environment = request.action === 'build' ? structuredClone(this.buildEnvironment()) : undefined;
    const environmentIdentity = JSON.stringify(environment);
    if (!cooker) throw new Error('ARC cooker is unavailable in the selected SDK');
    const toolStamp = () =>
      JSON.stringify(
        [
          tool,
          cooker,
          ...(environment
            ? [environment.cmake, path.join(environment.sdkRoot, 'lib', 'cmake', 'ARC', 'ARCConfig.cmake')]
            : []),
        ].map((executable) => {
          const stat = fs.statSync(executable);
          if (!stat.isFile()) throw new Error('Workflow executable is not a file');
          return { path: fs.realpathSync(executable), size: stat.size, modified: stat.mtimeMs };
        }),
      );
    const sdkIdentity = toolStamp();
    const descriptorPath = project.descriptorPath;
    const projectRoot = project.projectRoot;
    const runtimeConfig =
      request.action === 'build' ? path.resolve(projectRoot, project.descriptor.settings.renderer) : undefined;
    const runtimeConfigHash = () =>
      runtimeConfig ? createHash('sha256').update(fs.readFileSync(runtimeConfig)).digest('hex') : '';
    const capturedConfigHash = runtimeConfigHash();
    const descriptorHash = () => createHash('sha256').update(fs.readFileSync(descriptorPath)).digest('hex');
    const hash = descriptorHash();
    const descriptor = JSON.stringify(project.descriptor);
    const current = () => {
      try {
        const active = this.project();
        return (
          !this.cancelled &&
          active?.writable === true &&
          active.descriptorPath === descriptorPath &&
          active.projectRoot === projectRoot &&
          JSON.stringify(active.descriptor) === descriptor &&
          this.tool() === tool &&
          this.cooker(tool) === cooker &&
          toolStamp() === sdkIdentity &&
          descriptorHash() === hash &&
          runtimeConfigHash() === capturedConfigHash &&
          (request.action !== 'build' || JSON.stringify(this.buildEnvironment()) === environmentIdentity)
        );
      } catch {
        return false;
      }
    };
    this.busy = true;
    this.cancelled = false;
    this.runner = undefined;
    this.value = { requestId: randomUUID(), planning: true, run: null, output: [], error: null };
    this.emit();
    try {
      const plan = await queryProjectWorkflowPlan(tool, descriptorPath, request);
      if (!current() || plan.projectGuid !== project.descriptor.guid)
        throw new Error('Workflow cancelled or project/SDK changed while planning');
      const expectedStages = request.action === 'build' ? 'validate,configure,compile,cook,package' : 'validate,cook';
      if (plan.stages.join(',') !== expectedStages) throw new Error('Unexpected native workflow stages');
      const onOutput = (text: string) => {
        this.value.output.push(text);
        // Bounded diagnostics; each process chunk is at most 16 KiB.
        this.value.output = this.value.output.slice(-64);
        this.emit();
      };
      const adapter = createCookWorkflowAdapter(cooker, undefined, onOutput);
      const build =
        request.action === 'build'
          ? createWindowsBuildWorkflow(
              tool,
              cooker,
              environment,
              path.resolve(projectRoot, project.descriptor.paths.build),
              runtimeConfig!,
              undefined,
              onOutput,
            )
          : undefined;
      this.runner = new ProjectWorkflowRunner(
        build?.adapters ?? {
          validate: {
            capability: () => ({ available: true }),
            execute: async (context) => context.assertCurrent(),
            rollback: async () => {},
          },
          cook: adapter.adapter,
        },
        current,
        build?.publication ?? adapter.publication,
        (run) => {
          this.value.run = run;
          this.emit();
        },
      );
      this.value.planning = false;
      return await this.runner.execute(plan, {
        projectGuid: plan.projectGuid,
        projectRoot,
        descriptorPath,
        projectRevision: 0,
        sdkIdentity,
      });
    } catch (error) {
      this.value.error = error instanceof Error ? error.message : String(error);
      throw error;
    } finally {
      this.value.planning = false;
      this.busy = false;
      this.emit();
    }
  }
}
