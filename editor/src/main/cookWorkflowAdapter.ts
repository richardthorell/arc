import fs from 'node:fs';
import path from 'node:path';
import { z } from 'zod';
import type { WorkflowArtifact, WorkflowPlan } from '../common/projectWorkflowTypes';
import type { WorkflowStageAdapter, WorkflowPublication } from './projectWorkflowRunner';
import { runWorkflowProcess, type WorkflowProcessRequest } from './workflowProcess';

const manifestResult = z.object({ manifest: z.string().min(1) });
const completedCook = z.object({
  event: z.literal('cook.complete'),
  manifest: z.string().optional(),
  variants: z.array(manifestResult).min(1).max(16).optional(),
});
const manifestIdentity = z.object({
  buildId: z.string().min(1),
  target: z.object({
    name: z.string(),
    platform: z.string(),
    architecture: z.string(),
    renderer: z.string(),
    configuration: z.string(),
  }),
});

function contained(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative !== '' && !relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative);
}

/** A generation is immutable after verification. Only the small publication record is replaced. */
export function createCookWorkflowAdapter(
  cooker: string,
  runProcess: (request: WorkflowProcessRequest) => Promise<string> = runWorkflowProcess,
  onOutput?: (text: string) => void,
): { adapter: WorkflowStageAdapter; publication: WorkflowPublication } {
  let generation: string | undefined;
  let artifacts: WorkflowArtifact[] = [];
  let recordRoot: string | undefined;
  const supports = (plan: WorkflowPlan) => plan.action === 'cook' && plan.destination === 'none';
  return {
    adapter: {
      capability: (plan) =>
        !supports(plan)
          ? { available: false, reason: 'Only saved-content cooking is executable in this adapter' }
          : !['windows', 'linux', 'macos', 'android'].includes(plan.platform)
            ? { available: false, reason: 'Cooker does not implement this platform' }
            : cooker && fs.existsSync(cooker)
              ? { available: true }
              : { available: false, reason: 'ARC cooker is unavailable in the selected SDK' },
      execute: async (context) => {
        context.assertCurrent();
        if (!supports(context.plan)) throw new Error('Unsupported cook workflow');
        const projectRoot = fs.realpathSync(context.authority.projectRoot);
        const output = path.resolve(context.plan.cookOutput);
        if (!contained(projectRoot, output)) throw new Error('Cook output must be inside the project');
        // Verify existing ancestors before creating directories: never follow an escaped junction.
        let ancestor = output;
        while (!fs.existsSync(ancestor)) ancestor = path.dirname(ancestor);
        const realAncestor = fs.realpathSync(ancestor);
        if (realAncestor !== projectRoot && !contained(projectRoot, realAncestor))
          throw new Error('Cook output escapes through a symlink');
        fs.mkdirSync(output, { recursive: true });
        recordRoot = fs.realpathSync(output);
        if (!contained(projectRoot, recordRoot)) throw new Error('Cook output escapes project');
        generation = fs.mkdtempSync(path.join(recordRoot, 'generation-'));
        const text = await runProcess({
          executable: cooker,
          args: [
            'cook',
            '--project',
            context.authority.descriptorPath,
            '--profile',
            context.plan.profile,
            '--output',
            generation,
            '--json',
          ],
          cwd: projectRoot,
          signal: context.signal,
          onOutput,
        });
        context.assertCurrent();
        const completions = text.split(/\r?\n/u).flatMap((line) => {
          try {
            const parsed = completedCook.safeParse(JSON.parse(line));
            return parsed.success ? [parsed.data] : [];
          } catch {
            return [];
          }
        });
        if (completions.length !== 1) throw new Error('Cooker did not return exactly one completion record');
        const result = completions[0];
        const manifests =
          result.variants?.map((variant) => variant.manifest) ?? (result.manifest ? [result.manifest] : []);
        if (!manifests.length || new Set(manifests).size !== manifests.length)
          throw new Error('Cooker returned missing or duplicate manifests');
        artifacts = [];
        for (const manifest of manifests) {
          if (!path.isAbsolute(manifest) || !contained(generation, fs.realpathSync(manifest)))
            throw new Error('Cook manifest escaped its generation');
          if (fs.statSync(manifest).size > 16 * 1024 * 1024) throw new Error('Cook manifest exceeds metadata limit');
          const identity = manifestIdentity.parse(JSON.parse(fs.readFileSync(manifest, 'utf8')));
          const target = identity.target;
          if (
            target.name !== context.plan.profile ||
            target.platform !== context.plan.platform ||
            target.architecture !== context.plan.architecture ||
            target.renderer !== context.plan.renderer ||
            target.configuration !== (context.plan.configuration === 'Shipping' ? 'shipping' : 'development')
          )
            throw new Error('Cook manifest does not match the requested target/profile');
          await runProcess({
            executable: cooker,
            args: ['verify', '--project', context.authority.descriptorPath, '--manifest', manifest],
            cwd: projectRoot,
            signal: context.signal,
            onOutput,
          });
          context.assertCurrent();
          artifacts.push({
            kind: 'cooked',
            path: generation,
            manifestPath: manifest,
            buildId: identity.buildId,
            targetId: context.plan.targetId,
            profile: context.plan.profile,
          });
        }
      },
      // Retain failed private generations for diagnostics; they are never advertised as current.
      // Do not race a failed process-tree termination by deleting files still potentially in use.
      rollback: async () => {
        artifacts = [];
      },
    },
    publication: {
      commit: (context) => {
        context.assertCurrent();
        if (!recordRoot || !generation || !artifacts.length) throw new Error('No verified cook output to publish');
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
