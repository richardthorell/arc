import { describe, expect, it } from 'vitest';

import type { AiInstructionSkill, AiInstructionSourceSnapshot } from '../../../common/aiInstructionTypes';
import type { AiSkillCapability } from '../../../common/aiSkillTypes';
import type { AiRuntimeRequest } from '../../../common/aiRuntimeTypes';
import { resolveAiRuntimeInstructions } from './aiInstructionResolver';

const skill = (
  id: string,
  description: string,
  requiredCapabilities: AiSkillCapability[],
  options: Partial<AiInstructionSkill> = {},
): AiInstructionSkill => ({
  manifest: {
    format: 'arc-skill',
    formatVersion: 1,
    id,
    name: id,
    version: '1.0.0',
    description,
    requiredCapabilities,
    tools: [],
    contexts: ['scene'],
  },
  instructions: `Instructions for ${id}`,
  origin: 'builtin',
  projectGuid: null,
  ...options,
});

const request = (prompt: string): AiRuntimeRequest => ({
  conversationId: 'conversation',
  messages: [
    { id: 'arc-context:auto:scene', role: 'system', content: 'scene context' },
    { id: 'arc-context:auto:selection', role: 'system', content: 'selection context' },
    { id: 'user', role: 'user', content: prompt },
  ],
});

const sources = (skills: AiInstructionSkill[]): AiInstructionSourceSnapshot => ({
  revision: 1,
  projectGuid: 'project-guid',
  projectInstructions: 'Prefer metres and stable entity GUIDs.',
  skills,
  diagnostics: [],
});

describe('resolveAiRuntimeInstructions', () => {
  it('includes base/project instructions and only relevant capability-compatible skills', () => {
    const scene = skill('scene-inspection', 'Inspect scene entities, selection, components, and assets.', [
      'scene.read',
      'asset.read',
    ]);
    const material = skill('material-authoring', 'Create and edit materials and material graphs.', [
      'scene.read',
      'scene.mutate',
      'asset.read',
      'asset.mutate',
    ]);

    const resolution = resolveAiRuntimeInstructions(
      request('Inspect the selected scene entity'),
      sources([scene, material]),
    );

    expect(resolution.diagnostics.selectedSkillIds).toEqual(['scene-inspection']);
    expect(resolution.diagnostics.skills.find((entry) => entry.id === 'material-authoring')).toMatchObject({
      reason: 'missing-capability',
      missingCapabilities: ['scene.mutate', 'asset.mutate'],
    });
    expect(resolution.request.messages.slice(0, 3).map((message) => message.id)).toEqual([
      'arc-instructions:base',
      'arc-instructions:project',
      'arc-instructions:skill:scene-inspection',
    ]);
  });

  it('treats declared tools as advisory until the request actually exposes them', () => {
    const renderer = skill('renderer-diagnostics', 'Debug renderer viewport diagnostics.', [
      'viewport.read',
      'viewport.control',
      'diagnostics.read',
    ]);
    renderer.manifest.tools = ['viewport.move', 'diagnostics.get'];

    const withoutTool = resolveAiRuntimeInstructions(
      request('Debug renderer viewport diagnostics'),
      sources([renderer]),
    );
    expect(withoutTool.diagnostics.selectedSkillIds).toEqual([]);
    expect(withoutTool.diagnostics.skills[0]).toMatchObject({
      reason: 'missing-capability',
      missingCapabilities: ['viewport.control'],
      availableDeclaredTools: [],
    });

    const withTool = resolveAiRuntimeInstructions(
      {
        ...request('Debug renderer viewport diagnostics'),
        tools: [
          { name: 'viewport.move', description: 'Move the viewport', inputSchema: {} },
          { name: 'diagnostics.get', description: 'Read diagnostics', inputSchema: {} },
        ],
      },
      sources([renderer]),
    );
    expect(withTool.diagnostics.selectedSkillIds).toEqual(['renderer-diagnostics']);
    expect(withTool.diagnostics.skills[0].availableDeclaredTools).toEqual(['viewport.move', 'diagnostics.get']);
  });

  it('uses deterministic relevance ordering and does not inject unrelated skills', () => {
    const projectScene = skill('project-scene-review', 'Inspect scene entity hierarchy.', ['scene.read'], {
      origin: 'project',
      projectGuid: 'project-guid',
    });
    const builtinScene = skill('scene-inspection', 'Inspect scene entity hierarchy.', ['scene.read']);
    const play = skill('play-workflows', 'Run play mode and inspect runtime state.', ['scene.read', 'play.control']);

    const first = resolveAiRuntimeInstructions(
      request('Inspect the scene entity hierarchy'),
      sources([builtinScene, play, projectScene]),
    );
    const second = resolveAiRuntimeInstructions(
      request('Inspect the scene entity hierarchy'),
      sources([projectScene, builtinScene, play]),
    );

    expect(first.diagnostics.selectedSkillIds).toEqual(second.diagnostics.selectedSkillIds);
    expect(first.diagnostics.selectedSkillIds).toEqual(['project-scene-review', 'scene-inspection']);
    expect(first.diagnostics.selectedSkillIds).not.toContain('play-workflows');
  });
});
