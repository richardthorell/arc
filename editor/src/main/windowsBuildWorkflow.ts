import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { z } from 'zod';
import type { WorkflowArtifact, WorkflowCapability, WorkflowPlan } from '../common/projectWorkflowTypes';
import type { WorkflowStageAdapter, WorkflowStageContext, WorkflowPublication } from './projectWorkflowRunner';
import type { WorkflowProcessRequest } from './workflowProcess';
import { runWorkflowProcess } from './workflowProcess';
import { createCookWorkflowAdapter } from './cookWorkflowAdapter';
import type { EditorSettingsSnapshot } from '../common/editorWorkflowTypes';
import { windowsToolchainSettingKeys, resolveVisualStudioGenerator } from './windowsToolchain';

/** Trusted main-process configuration, never renderer-supplied command arguments. */
export type WindowsBuildEnvironment = {
  cmake: string;
  sdkRoot: string;
  visualStudio: string;
  generator: string;
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
    const sdkRoot = path.dirname(path.dirname(fs.realpathSync(tool)));
    if (!fs.statSync(path.join(sdkRoot, 'lib', 'cmake', 'ARC', 'ARCConfig.cmake')).isFile()) return undefined;
    const generator = resolveVisualStudioGenerator(studio.resolvedPath);
    if (!generator) return undefined;
    return { cmake: cmake.resolvedPath, sdkRoot, visualStudio: studio.resolvedPath, generator };
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
): {
  adapters: Partial<Record<'validate' | 'configure' | 'compile' | 'cook' | 'package', WorkflowStageAdapter>>;
  publication: WorkflowPublication;
} {
  const cook = createCookWorkflowAdapter(cooker, runProcess, onOutput);
  let codeDirectory = '';
  let buildManifest = '';
  let generation = '';
  let recordRoot = '';
  let artifacts: WorkflowArtifact[] = [];
  const command = async (context: WorkflowStageContext, executable: string, args: string[]) => {
    context.assertCurrent();
    const result = await runProcess({
      executable,
      args,
      cwd: context.authority.projectRoot,
      signal: context.signal,
      onOutput,
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
          `-DARC_WORKFLOW_PROFILE=${context.plan.profile}`,
          `-DCMAKE_CONFIGURATION_TYPES=${context.plan.configuration}`,
        ]);
      }),
      compile: stage(async (context) => {
        if (!environment) throw new Error('Missing runtime build environment');
        await command(context, environment.cmake, [
          '--build',
          codeDirectory,
          '--config',
          context.plan.configuration,
          '--target',
          'arc-runtime-output',
          '--parallel',
        ]);
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
