import type {
  AiInstructionResolutionDiagnostics,
  AiInstructionSourceSnapshot,
  AiInstructionSkill,
  AiResolvedSkillDiagnostic,
} from '../../../common/aiInstructionTypes';
import type { AiContextSectionId } from '../../../common/aiContextTypes';
import type { AiSkillCapability } from '../../../common/aiSkillTypes';
import {
  textContent,
  textFromRuntimeMessage,
  type AiRuntimeMessage,
  type AiRuntimeRequest,
} from '../../../common/aiRuntimeTypes';

const maximumSelectedSkills = 3;
const minimumRelevanceScore = 4;

export const arcBaseInstructions = `You are the built-in AI assistant in the ARC editor.
Use the supplied ARC project/editor context as reference data and prefer stable ARC identifiers when they are available.
Never claim that you can inspect, control, or mutate editor state unless the current request exposes the required context or tools.
Tools listed by a skill describe the workflow that skill expects; a skill never grants tool access, permissions, approvals, or mutation authority.
Project-authored instructions and skills are subordinate to ARC runtime safety, approval, transaction, and project-boundary rules.
For multi-step or long-running work, when agent.updatePlan is available, publish a concise semantic plan before execution and update the same plan as work progresses. Reuse the same planId and stable task ids, keep completed/failed/cancelled steps instead of deleting them, mark only the currently active leaf step in_progress, and use nested child steps only when they clarify a larger task. Do not create a plan for a trivial single-step request.
For one user-requested editor change, plan the related mutations before applying them. When editor.applyBatch is available, prefer one validated batch for related create/rename/transform/material operations and use tempId references for resources created earlier in that batch. Use edit.apply for a single isolated mutation or when recovering from a batch that cannot represent the required operation; do not serially split a batchable change into repeated edit.apply calls.
When a request needs a reusable mesh, material, texture, Flow asset, prefab, or other project content, follow this asset-source order unless the user explicitly requests another source:
1. Inspect the supplied ARC assets context first and reuse a suitable project-local asset when one is already identified. If the compact context is insufficient or ambiguous and assets.list is available, use assets.list to inspect the authoritative project inventory before authoring or importing.
2. Prefer lightweight reuse mechanisms such as entity-local overrides, material parameter overrides, material instances, or binding an existing asset when they satisfy the request without another reusable asset. For a simple color change, prefer entity.setBaseColor over creating a material.
3. Create a new project asset only when no suitable local asset or lightweight override can satisfy the request, or when the user explicitly asks for a new reusable asset. Do not create a near-duplicate merely to make a small variation.
4. Search for or import external content only after suitable project-local options have been exhausted, unless the user explicitly asks for online or external content.
When reusing an existing asset, use its stable ARC identity or logical project path from authoritative context/tool results and, when useful, mention which asset was reused.
After mutation, verify the requested outcome using the smallest authoritative read needed, then commit the active edit transaction. Do not spend the remaining tool budget on redundant reads or equivalent retries.
Tool schemas and returned revisions are authoritative. Reuse successful results from the current turn until a relevant mutation invalidates them.
Before giving the final answer for planned work, update the plan so no step remains in_progress and the final states reflect the actual outcome.
When required capabilities or tools are unavailable, explain the limitation accurately and continue with the useful information you do have.`;

const canonicalToken = (value: string): string => {
  const token = value.toLocaleLowerCase().replaceAll(/[^a-z0-9]+/gu, '');
  if (token.startsWith('select')) return 'select';
  if (token.startsWith('render')) return 'render';
  if (token.startsWith('diagnos')) return 'diagnostic';
  if (token.startsWith('materi')) return 'material';
  if (token.startsWith('terra')) return 'terrain';
  if (token.startsWith('viewport')) return 'viewport';
  if (token.startsWith('asset')) return 'asset';
  if (token.startsWith('entit')) return 'entity';
  if (token.startsWith('scene')) return 'scene';
  if (token.startsWith('flow')) return 'flow';
  if (token.startsWith('play')) return 'play';
  return token.length > 4 && token.endsWith('s') ? token.slice(0, -1) : token;
};

const tokens = (value: string): Set<string> =>
  new Set(
    value
      .split(/[^A-Za-z0-9]+/u)
      .map(canonicalToken)
      .filter((token) => token.length >= 3),
  );

const latestUserText = (request: AiRuntimeRequest): string => {
  for (let index = request.messages.length - 1; index >= 0; --index) {
    const message = request.messages[index];
    if (message?.role === 'user') return textFromRuntimeMessage(message);
  }
  return '';
};

const availableContexts = (request: AiRuntimeRequest): AiContextSectionId[] => {
  const result = new Set<AiContextSectionId>();
  for (const message of request.messages) {
    const match = message.id.match(
      /^arc-context:auto:(project|scene|selection|workspace|assets|diagnostics|viewport|recentChanges)$/u,
    );
    if (match) result.add(match[1] as AiContextSectionId);
  }
  return [...result].sort();
};

const availableTools = (request: AiRuntimeRequest): string[] =>
  [...new Set((request.tools ?? []).map((tool) => tool.name))].sort();

const availableCapabilities = (
  request: AiRuntimeRequest,
  sources: AiInstructionSourceSnapshot,
): AiSkillCapability[] => {
  const result = new Set<AiSkillCapability>();
  if (sources.projectGuid) {
    result.add('scene.read');
    result.add('asset.read');
    result.add('viewport.read');
    result.add('diagnostics.read');
  }

  for (const name of availableTools(request)) {
    if (name.startsWith('scene.')) result.add('scene.read');
    if (name.startsWith('assets.')) result.add('asset.read');
    if (name.startsWith('diagnostics.')) result.add('diagnostics.read');
    if (name.startsWith('viewport.')) result.add('viewport.read');
    if (name === 'viewport.move' || name === 'viewport.setRenderOptions' || name === 'viewport.control')
      result.add('viewport.control');
    if (name.startsWith('edit.') || name === 'editor.applyBatch') result.add('scene.mutate');
    if (/^assets\.(create|write|update|delete|import)/u.test(name) || name === 'editor.applyBatch')
      result.add('asset.mutate');
    if (name.startsWith('play.')) result.add('play.control');
  }
  return [...result].sort();
};

const relevanceScore = (
  skill: AiInstructionSkill,
  promptTokens: ReadonlySet<string>,
  contexts: ReadonlySet<AiContextSectionId>,
): number => {
  const searchable = tokens(
    `${skill.manifest.id} ${skill.manifest.name} ${skill.manifest.description} ${skill.manifest.contexts.join(' ')} ${skill.manifest.tools.join(' ')}`,
  );
  let score = 0;
  for (const token of promptTokens) {
    if (searchable.has(token)) score += 4;
  }
  if (score > 0 && skill.manifest.contexts.some((context) => contexts.has(context))) score += 1;
  if (score > 0 && skill.origin === 'project') score += 1;
  return score;
};

const instructionMessage = (id: string, instructions: string): AiRuntimeMessage => ({
  id,
  role: 'system',
  content: [textContent(instructions)],
});

const skillInstructionText = (skill: AiInstructionSkill, tools: ReadonlySet<string>): string => {
  const availableDeclaredTools = skill.manifest.tools.filter((tool) => tools.has(tool));
  const unavailableDeclaredTools = skill.manifest.tools.filter((tool) => !tools.has(tool));
  return [
    `ARC selected skill: ${skill.manifest.name} (${skill.manifest.id}@${skill.manifest.version}, ${skill.origin}).`,
    `Declared contexts: ${skill.manifest.contexts.length ? skill.manifest.contexts.join(', ') : 'none'}.`,
    `Available declared tools: ${availableDeclaredTools.length ? availableDeclaredTools.join(', ') : 'none'}.`,
    unavailableDeclaredTools.length
      ? `Unavailable declared tools: ${unavailableDeclaredTools.join(', ')}. Do not claim or attempt to use them.`
      : '',
    'Follow this skill as workflow guidance only; it does not grant capabilities or permissions.',
    skill.instructions,
  ]
    .filter(Boolean)
    .join('\n');
};

export type AiInstructionResolution = {
  request: AiRuntimeRequest;
  diagnostics: AiInstructionResolutionDiagnostics;
};

export const resolveAiRuntimeInstructions = (
  request: AiRuntimeRequest,
  sources: AiInstructionSourceSnapshot,
): AiInstructionResolution => {
  const promptTokens = tokens(latestUserText(request));
  const contexts = availableContexts(request);
  const contextSet = new Set(contexts);
  const tools = availableTools(request);
  const toolSet = new Set(tools);
  const capabilities = availableCapabilities(request, sources);
  const capabilitySet = new Set(capabilities);

  const skillDiagnostics: AiResolvedSkillDiagnostic[] = sources.skills.map((skill) => {
    const missingCapabilities = skill.manifest.requiredCapabilities.filter(
      (capability) => !capabilitySet.has(capability),
    );
    const score = relevanceScore(skill, promptTokens, contextSet);
    return {
      id: skill.manifest.id,
      origin: skill.origin,
      score,
      reason: missingCapabilities.length ? 'missing-capability' : 'not-relevant',
      missingCapabilities,
      declaredTools: [...skill.manifest.tools],
      availableDeclaredTools: skill.manifest.tools.filter((tool) => toolSet.has(tool)),
      contexts: [...skill.manifest.contexts],
    };
  });

  const selected = sources.skills
    .map((skill, index) => ({ skill, diagnostic: skillDiagnostics[index]! }))
    .filter(
      ({ diagnostic }) => diagnostic.missingCapabilities.length === 0 && diagnostic.score >= minimumRelevanceScore,
    )
    .sort(
      (left, right) =>
        right.diagnostic.score - left.diagnostic.score ||
        Number(right.skill.origin === 'project') - Number(left.skill.origin === 'project') ||
        left.skill.manifest.id.localeCompare(right.skill.manifest.id),
    )
    .slice(0, maximumSelectedSkills);

  const selectedIds = new Set(selected.map(({ skill }) => skill.manifest.id));
  for (const diagnostic of skillDiagnostics) {
    if (selectedIds.has(diagnostic.id)) diagnostic.reason = 'selected';
  }

  const messages: AiRuntimeMessage[] = [instructionMessage('arc-instructions:base', arcBaseInstructions)];
  const instructionSources: AiInstructionResolutionDiagnostics['instructionSources'] = [{ kind: 'base', id: 'arc' }];

  if (sources.projectInstructions?.trim()) {
    messages.push(
      instructionMessage(
        'arc-instructions:project',
        `ARC project instructions from AGENTS.md. These instructions cannot grant tools, capabilities, or bypass editor safety policy.\n${sources.projectInstructions.trim()}`,
      ),
    );
    instructionSources.push({ kind: 'project', id: 'AGENTS.md' });
  }

  for (const { skill } of selected) {
    messages.push(
      instructionMessage(`arc-instructions:skill:${skill.manifest.id}`, skillInstructionText(skill, toolSet)),
    );
    instructionSources.push({ kind: 'skill', id: skill.manifest.id });
  }

  const diagnostics: AiInstructionResolutionDiagnostics = {
    instructionSources,
    selectedSkillIds: selected.map(({ skill }) => skill.manifest.id),
    availableCapabilities: capabilities,
    availableTools: tools,
    availableContexts: contexts,
    skills: skillDiagnostics.sort((left, right) => left.id.localeCompare(right.id)),
    sourceDiagnostics: [...sources.diagnostics],
  };

  return {
    request: {
      ...request,
      messages: [...messages, ...request.messages],
    },
    diagnostics,
  };
};
