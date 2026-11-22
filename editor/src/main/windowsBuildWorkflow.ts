import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { z } from 'zod';
import type { BuildTargetActivePhase, BuildTargetGraph } from '../common/buildTargetTypes';
import type { WorkflowArtifact, WorkflowCapability, WorkflowPlan } from '../common/projectWorkflowTypes';
import type { WorkflowStageAdapter, WorkflowStageContext, WorkflowPublication } from './projectWorkflowRunner';
import type { WorkflowProcessRequest } from './workflowProcess';
import { runWorkflowProcess } from './workflowProcess';
import { createCookWorkflowAdapter } from './cookWorkflowAdapter';
import { CMakeTargetOutputObserver, readCMakeTargetGraph, requestCMakeCodemodel } from './cmakeTargetGraph';
import type { EditorSettingsSnapshot } from '../common/editorWorkflowTypes';
import { windowsToolchainSettingKeys, resolveVisualStudioGenerator } from './windowsToolchain';
import { beginMsbuildCompileTraceForBuild } from './msbuildCompileTasks';

/** Trusted main-process configuration, never renderer-supplied command arguments. */
export type WindowsBuildEnvironment = {
  cmake: string;
  sdkRoot: string;
  visualStudio: string;
  generator: string;
  development?: { buildDirectory: string; engineVersion: string; architecture: string };
};

export function resolveWindowsBuildEnvironment(
  tool: string,
  settings: EditorSettingsSnapshot | null,
): WindowsBuildEnvironment | undefined {
  try {
    const cmake = settings?.pathValidation?.[windowsToolchainSettingKeys.cmakePath];
    const studio = settings?.pathValidation?.[windowsToolchainSettingKeys.visualStudioPath];
    if (!cmake?.valid || !studio?.valid) return undefined;
    // An installed CLI's sibling SDK is authoritative; a development-tree CLI
    // cannot silently select some other registered SDK or a newest build folder.
    let sdkRoot = path.dirname(path.dirname(fs.realpathSync(tool)));
    let development: WindowsBuildEnvironment['development'];
    if (process.env.ARC_DEVELOPMENT_SDK) {
      sdkRoot = fs.realpathSync(process.env.ARC_DEVELOPMENT_SDK);
      const configuration = process.env.ARC_DEVELOPMENT_CONFIGURATION;
      if (!configuration || !/^[A-Za-z0-9_-]+$/.test(configuration)) return undefined;
      const metadata = z
        .object({
          version: z.literal(1),
          engineVersion: z.string(),
          buildDirectory: z.string(),
          projectTool: z.string(),
          architecture: z.string(),
        })
        .parse(readJson(path.join(sdkRoot, configuration, 'arc-development-sdk.json')));
      if (fs.realpathSync(metadata.projectTool) !== fs.realpathSync(tool)) return undefined;
      development = {
        buildDirectory: metadata.buildDirectory,
        engineVersion: metadata.engineVersion,
        architecture: metadata.architecture,
      };
      if (!fs.statSync(path.join(sdkRoot, configuration, 'lib', 'cmake', 'ARC', 'ARCConfig.cmake')).isFile())
        return undefined;
      // Runtime output currently uses Shipping; prepare that configuration on demand.
      sdkRoot = path.join(sdkRoot, 'Shipping');
    }
    if (!development && !fs.statSync(path.join(sdkRoot, 'lib', 'cmake', 'ARC', 'ARCConfig.cmake')).isFile())
      return undefined;
    const generator = resolveVisualStudioGenerator(studio.resolvedPath);
    if (!generator) return undefined;
    return {
      cmake: cmake.resolvedPath,
      sdkRoot,
      visualStudio: studio.resolvedPath,
      generator,
      ...(development ? { development } : {}),
    };
  } catch {
    return undefined;
  }
}

function inside(root: string, file: string): boolean {
  const relative = path.relative(root, file);
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function outputDirectory(root: string, output: string): string {
  root = fs.realpathSync(root);
  output = path.resolve(output);
  if (!inside(root, output)) throw new Error('Workflow output must be inside the project');
  let ancestor = output;
  while (!fs.existsSync(ancestor)) ancestor = path.dirname(ancestor);
  const real = fs.realpathSync(ancestor);
  if (real !== root && !inside(root, real)) throw new Error('Workflow output escapes through a symlink');
  fs.mkdirSync(output, { recursive: true });
  const resolved = fs.realpathSync(output);
  if (!inside(root, resolved)) throw new Error('Workflow output escapes project');
  return resolved;
}

function readJson(file: string): unknown {
  if (fs.statSync(file).size > 16 * 1024 * 1024) throw new Error('Workflow metadata exceeds size limit');
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

/** Bounded parser for explicit MSBuild project artifact lines. */
export class SdkModuleOutputObserver {
  private pendingStdout = '';
  private pendingStderr = '';
  private readonly projectsByContext = new Map<string, string>();

  constructor(
    private readonly buildDirectory: string,
    private readonly onModule: (name: string, artifact: string) => void,
    private readonly onPhase?: (name: string, phase: BuildTargetActivePhase) => void,
  ) {}

  accept(text: string, stream: 'stdout' | 'stderr'): void {
    const combined = (stream === 'stdout' ? this.pendingStdout : this.pendingStderr) + text;
    const lines = combined.split(/\r?\n/u);
    const rest = (lines.pop() ?? '').slice(-8192);
    if (stream === 'stdout') this.pendingStdout = rest;
    else this.pendingStderr = rest;
    for (const line of lines) {
      this.observePhase(line);
      const arrow = line.lastIndexOf(' -> ');
      if (arrow < 0) continue;
      const project = /([A-Za-z0-9_.+-]{1,128})\.vcxproj\s*$/iu.exec(line.slice(0, arrow));
      if (!project || project[1].toLowerCase() === 'zero_check') continue;
      const raw = line
        .slice(arrow + 4)
        .trim()
        .replace(/^["']|["']$/gu, '');
      if (!raw || raw.length > 2048 || /[\0\r\n]/u.test(raw)) continue;
      const artifact = path.resolve(this.buildDirectory, raw);
      if (!inside(this.buildDirectory, artifact)) continue;
      this.onModule(project[1], artifact);
    }
  }

  private observePhase(line: string): void {
    const prefix = /^\s*(\d+)>/u.exec(line)?.[1];
    const projects = [...line.matchAll(/"([^"]+\.vcxproj)"(?:\s+\((\d+)\))?/giu)];
    if (projects.length) {
      const project = projects[projects.length - 1];
      const name = path.basename(project[1], path.extname(project[1]));
      const context = project[2] ?? prefix;
      if (context && name.toLowerCase() !== 'zero_check') this.projectsByContext.set(context, name);
    }
    if (!prefix || !this.onPhase) return;
    const task = /^\s*\d+>\s*(ClCompile|Link|Lib):/iu.exec(line)?.[1]?.toLowerCase();
    if (!task) return;
    const name = this.projectsByContext.get(prefix);
    if (!name) return;
    this.onPhase(name, task === 'clcompile' ? 'compiling' : 'linking');
  }
}

/** Reads the generated MSBuild project-reference graph without scanning for binaries. */
export function readMsbuildProjectGraph(
  buildDirectory: string,
  configuration: string,
  rootTarget: string,
): BuildTargetGraph | null {
  try {
    const root = fs.realpathSync(buildDirectory);
    const rootProject = path.join(root, `${rootTarget}.vcxproj`);
    if (!fs.statSync(rootProject).isFile() || fs.lstatSync(rootProject).isSymbolicLink()) return null;
    const id = (filename: string) => path.relative(root, filename).replace(/\\/gu, '/').toLowerCase();
    const pending = [rootProject];
    const visited = new Set<string>();
    const nodes: BuildTargetGraph['nodes'] = [];
    let truncated = false;
    while (pending.length) {
      const filename = pending.shift()!;
      const nodeId = id(filename);
      if (visited.has(nodeId)) continue;
      if (visited.size >= 256) {
        truncated = true;
        break;
      }
      visited.add(nodeId);
      const info = fs.lstatSync(filename);
      if (!info.isFile() || info.isSymbolicLink() || info.size > 4 * 1024 * 1024) continue;
      const content = fs.readFileSync(filename, 'utf8');
      const dependencies: string[] = [];
      const references = /<ProjectReference\b[^>]*\bInclude="([^"]+)"/giu;
      for (const match of content.matchAll(references)) {
        const include = match[1].replace(/&amp;/giu, '&');
        if (!include || /[\0\r\n]/u.test(include) || path.extname(include).toLowerCase() !== '.vcxproj') continue;
        // Generated Visual Studio projects use Windows separators even when a
        // fixture is inspected on another host (for example Linux CI).
        const resolved = path.resolve(path.dirname(filename), include.replace(/[\\/]/gu, path.sep));
        if (!inside(root, resolved) || !fs.existsSync(resolved)) continue;
        if (path.basename(resolved, path.extname(resolved)).toLowerCase() === 'zero_check') continue;
        const dependencyId = id(resolved);
        if (!dependencies.includes(dependencyId)) dependencies.push(dependencyId);
        if (!visited.has(dependencyId)) pending.push(resolved);
      }
      nodes.push({
        id: nodeId,
        name: path.basename(filename, path.extname(filename)),
        type: 'MSBUILD_PROJECT',
        dependencies,
        artifacts: [],
      });
    }
    return {
      revision: 1,
      generator: 'MSBuild',
      configuration,
      rootTargetId: id(rootProject),
      discoveredAt: new Date().toISOString(),
      nodes,
      truncated,
    };
  } catch {
    return null;
  }
}

export function windowsBuildCapability(
  plan: WorkflowPlan,
  environment: WindowsBuildEnvironment | undefined,
  host = process.platform,
): WorkflowCapability {
  if (host !== 'win32' || plan.platform !== 'windows' || plan.configuration !== 'Shipping')
    return { available: false, reason: 'Runtime Build currently requires Windows Shipping on a Windows host' };
  if (plan.action !== 'build' || plan.destination !== 'none' || plan.sessionId)
    return { available: false, reason: 'This adapter builds saved output only; it never launches a runtime' };
  if (!['x86_64', 'arm64'].includes(plan.architecture))
    return { available: false, reason: 'Unsupported Windows runtime architecture' };
  if (!environment)
    return {
      available: false,
      reason: 'Configure CMake and Visual Studio and use an installed ARC SDK to build runtime output',
    };
  if (environment.development && environment.development.engineVersion !== plan.engineVersion)
    return { available: false, reason: 'The project requires a different engine version than the development SDK' };
  if (environment.development && environment.development.architecture !== plan.architecture)
    return { available: false, reason: 'The selected target architecture differs from the development SDK' };
  return { available: true };
}

/** Native planning selects stages; this executor does not own packaging policy. */
export function createWindowsBuildWorkflow(
  tool: string,
  cooker: string,
  environment: WindowsBuildEnvironment | undefined,
  buildRoot: string,
  runtimeConfig: string,
  runProcess: (request: WorkflowProcessRequest) => Promise<string> = runWorkflowProcess,
  onOutput?: (text: string) => void,
  host = process.platform,
  onRawOutput?: (text: string, stream: 'stdout' | 'stderr') => void,
): {
  adapters: Partial<Record<'validate' | 'configure' | 'compile' | 'cook' | 'package', WorkflowStageAdapter>>;
  publication: WorkflowPublication;
} {
  const cook = createCookWorkflowAdapter(cooker, runProcess, onOutput, onRawOutput);
  let codeDirectory = '';
  let buildManifest = '';
  let targetGraph: ReturnType<typeof readCMakeTargetGraph> = null;
  let generation = '';
  let recordRoot = '';
  let artifacts: WorkflowArtifact[] = [];
  const command = async (
    context: WorkflowStageContext,
    executable: string,
    args: string[],
    observeRaw?: (text: string, stream: 'stdout' | 'stderr') => void,
  ) => {
    context.assertCurrent();
    const result = await runProcess({
      executable,
      args,
      cwd: context.authority.projectRoot,
      signal: context.signal,
      onOutput,
      onRawOutput: (text, stream) => {
        onRawOutput?.(text, stream);
        observeRaw?.(text, stream);
      },
      outputMode: executable === environment?.cmake ? 'log' : 'protocol',
    });
    context.assertCurrent();
    return result;
  };
  const stage = (execute: WorkflowStageAdapter['execute']): WorkflowStageAdapter => ({
    capability: (plan) => windowsBuildCapability(plan, environment, host),
    execute,
    // Private generations are retained for diagnosis, never advertised after failure.
    rollback: async () => {
      artifacts = [];
    },
  });
  return {
    adapters: {
      validate: stage(async (context) => {
        context.assertCurrent();
        const projectRoot = fs.realpathSync(context.authority.projectRoot);
        if (!inside(projectRoot, fs.realpathSync(runtimeConfig)))
          throw new Error('Runtime configuration must belong to the project');
        if (!fs.statSync(path.join(projectRoot, 'CMakeLists.txt')).isFile())
          throw new Error('Project has no CMake runtime build');
        // Hash the captured identity rather than using profile text as a filesystem path.
        const key = createHash('sha256')
          .update(
            JSON.stringify([context.plan.targetId, context.plan.profile, context.plan.configuration, environment]),
          )
          .digest('hex')
          .slice(0, 24);
        codeDirectory = outputDirectory(projectRoot, path.join(buildRoot, 'RuntimeCode', key));
        buildManifest = path.join(codeDirectory, `arc-runtime-${context.plan.configuration}.json`);
      }),
      configure: stage(async (context) => {
        if (!environment) throw new Error('Missing runtime build environment');
        if (environment.development) {
          await context.runStageTask('development-sdk', 'Build SDK', async (progress) => {
            let sdkGraph: ReturnType<typeof readCMakeTargetGraph> = null;
            try {
              sdkGraph = readCMakeTargetGraph(
                environment.development!.buildDirectory,
                context.plan.configuration,
                'arc-development-sdk',
              );
            } catch {
              // Explicit MSBuild artifact lines still populate bounded module rows below.
            }
            progress.publishTargetGraph(
              sdkGraph ??
                readMsbuildProjectGraph(
                  environment.development!.buildDirectory,
                  context.plan.configuration,
                  'arc-development-sdk',
                ) ?? {
                  revision: 1,
                  generator: environment.generator,
                  configuration: context.plan.configuration,
                  rootTargetId: 'arc-development-sdk',
                  discoveredAt: new Date().toISOString(),
                  truncated: false,
                  nodes: [
                    {
                      id: 'arc-development-sdk',
                      name: 'arc-development-sdk',
                      type: 'UTILITY',
                      dependencies: [],
                      artifacts: [],
                    },
                  ],
                },
            );
            const modules = new SdkModuleOutputObserver(
              environment.development!.buildDirectory,
              (name, artifact) => progress.reportTargetByName(name, artifact),
              (name, phase) => progress.reportTargetPhaseByName(name, phase),
            );
            await command(
              context,
              environment.cmake,
              [
                '--build',
                environment.development!.buildDirectory,
                '--config',
                context.plan.configuration,
                '--target',
                'arc-development-sdk',
                '--',
                '/verbosity:normal',
              ],
              (text, stream) => modules.accept(text, stream),
            );
            progress.reportRequestedTargetSatisfied();
          });
        }
        await context.runStageTask('cmake-configure', 'Configure project', async () => {
          // CMake responds to File API queries as part of this configure step.
          // Query failure must not turn a valid native build into an editor failure.
          try {
            requestCMakeCodemodel(codeDirectory);
          } catch {
            // The build remains authoritative even when diagnostics are unavailable.
          }
          await command(context, environment.cmake, [
            '-S',
            context.authority.projectRoot,
            '-B',
            codeDirectory,
            '-G',
            environment.generator,
            '-A',
            context.plan.architecture === 'arm64' ? 'ARM64' : 'x64',
            `-DCMAKE_GENERATOR_INSTANCE=${environment.visualStudio}`,
            `-DCMAKE_PREFIX_PATH=${environment.sdkRoot}`,
            `-DARC_DIR=${path.join(environment.sdkRoot, 'lib', 'cmake', 'ARC')}`,
            `-DARC_WORKFLOW_PROFILE=${context.plan.profile}`,
            `-DARC_WORKFLOW_PROJECT_GUID=${context.plan.projectGuid}`,
            `-DARC_WORKFLOW_ENGINE_VERSION=${context.plan.engineVersion}`,
            `-DARC_WORKFLOW_RENDERER=${context.plan.renderer}`,
            '-DARC_WORKFLOW_ROLE=client',
            `-DCMAKE_CONFIGURATION_TYPES=${context.plan.configuration}`,
          ]);
          try {
            targetGraph = readCMakeTargetGraph(codeDirectory, context.plan.configuration, 'arc-runtime-output');
            if (targetGraph) context.publishTargetGraph(targetGraph);
          } catch {
            // Missing, malformed, or unsupported File API metadata is nonfatal.
          }
        });
      }),
      compile: stage(async (context) => {
        if (!environment) throw new Error('Missing runtime build environment');
        const compileCapture = beginMsbuildCompileTraceForBuild(
          context.authority.projectRoot,
          codeDirectory,
          context.runId,
          host,
        );
        const observer = targetGraph
          ? new CMakeTargetOutputObserver(codeDirectory, targetGraph, (id) => context.reportTargetArtifact(id))
          : undefined;
        let compilePoll: ReturnType<typeof setInterval> | undefined;
        const publishTrace = (completed: boolean) => {
          const trace = completed ? compileCapture?.finish() : compileCapture?.snapshot();
          if (trace && (completed || trace.tasks.length || trace.activeTasks?.length))
            context.publishCompileTrace(trace);
        };
        if (compileCapture)
          compilePoll = setInterval(() => {
            try {
              publishTrace(false);
            } catch {
              // Native timing is best effort and never changes build results.
            }
          }, 750);
        try {
          await command(
            context,
            environment.cmake,
            [
              '--build',
              codeDirectory,
              '--config',
              context.plan.configuration,
              '--target',
              'arc-runtime-output',
              '--parallel',
              ...(compileCapture?.nativeArgs ? ['--', ...compileCapture.nativeArgs] : []),
            ],
            (text, stream) => observer?.accept(text, stream),
          );
        } finally {
          if (compilePoll !== undefined) clearInterval(compilePoll);
          try {
            publishTrace(true);
          } catch {
            // Native timing is best effort and never changes build results.
          }
        }
        // A successful native build satisfies the explicitly requested root,
        // even if incremental dependencies emitted no artifact output.
        context.reportRequestedTargetSatisfied();
        const result = JSON.parse(
          await command(context, tool, [
            'package',
            '--json',
            '--verify-build-only',
            '--project',
            context.authority.descriptorPath,
            '--profile',
            context.plan.profile,
            '--build-manifest',
            buildManifest,
          ]),
        );
        z.object({ succeeded: z.literal(true), executable: z.string().min(1) }).parse(result);
      }),
      cook: cook.adapter,
      package: stage(async (context) => {
        const cooked = cook.prepared();
        if (cooked.length !== 1)
          throw new Error(
            'Windows packaging requires exactly one cooked texture variant; select a single-output profile',
          );
        recordRoot = outputDirectory(context.authority.projectRoot, context.plan.packageOutput);
        // The native assembler publishes into a fresh private directory, not the current package.
        generation = fs.mkdtempSync(path.join(recordRoot, 'generation-'));
        const contentRoot = path.join(generation, 'content');
        const packed = z
          .object({ event: z.literal('package.complete'), manifest: z.string().min(1) })
          .parse(
            JSON.parse(
              await command(context, cooker, [
                'package',
                '--json',
                '--project',
                context.authority.descriptorPath,
                '--profile',
                context.plan.profile,
                '--manifest',
                cooked[0].manifestPath,
                '--output',
                contentRoot,
              ]),
            ),
          );
        if (!path.isAbsolute(packed.manifest) || !inside(contentRoot, fs.realpathSync(packed.manifest)))
          throw new Error('Content package escaped its private generation');
        await command(context, cooker, [
          'verify',
          '--project',
          context.authority.descriptorPath,
          '--manifest',
          packed.manifest,
        ]);
        const output = path.join(generation, 'output');
        const result = z
          .object({ succeeded: z.literal(true), output: z.string(), manifest: z.string() })
          .parse(
            JSON.parse(
              await command(context, tool, [
                'package',
                '--json',
                '--project',
                context.authority.descriptorPath,
                '--profile',
                context.plan.profile,
                '--build-manifest',
                buildManifest,
                '--cook-manifest',
                packed.manifest,
                '--runtime-config',
                runtimeConfig,
                '--output',
                output,
              ]),
            ),
          );
        if (
          path.resolve(result.output) !== output ||
          path.resolve(result.manifest) !== path.join(output, 'arc-distribution.json') ||
          !inside(generation, fs.realpathSync(result.manifest))
        )
          throw new Error('Packager returned output outside its private generation');
        const manifest = z
          .object({
            projectGuid: z.string(),
            target: z.object({ id: z.string(), cookProfile: z.string(), configuration: z.string() }),
          })
          .parse(readJson(result.manifest));
        if (
          manifest.projectGuid !== context.plan.projectGuid ||
          manifest.target.id !== context.plan.targetId ||
          manifest.target.cookProfile !== context.plan.profile ||
          manifest.target.configuration !== context.plan.configuration
        )
          throw new Error('Packaged identity does not match the captured workflow');
        artifacts = [
          {
            kind: 'package',
            path: output,
            manifestPath: result.manifest,
            buildId: createHash('sha256').update(fs.readFileSync(result.manifest)).digest('hex'),
            targetId: context.plan.targetId,
            profile: context.plan.profile,
          },
        ];
      }),
    },
    publication: {
      commit: (context) => {
        context.assertCurrent();
        if (!artifacts.length || !generation) throw new Error('No verified package to publish');
        const temporary = path.join(generation, 'publication.json');
        fs.writeFileSync(
          temporary,
          JSON.stringify(
            {
              version: 1,
              projectGuid: context.plan.projectGuid,
              sdkIdentity: context.authority.sdkIdentity,
              runId: context.runId,
              artifacts,
            },
            null,
            2,
          ),
          { flag: 'wx' },
        );
        context.assertCurrent();
        fs.renameSync(temporary, path.join(recordRoot, 'current-workflow.json'));
        return structuredClone(artifacts);
      },
    },
  };
}
