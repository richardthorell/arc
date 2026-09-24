import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { AiProviderAccountsSnapshot, AiProviderId } from '../common/aiProviderTypes';
import type { ArcProjectCandidate } from '../common/projectTypes';
import { SettingsService, type SettingsAiProviderService } from './settingsService';

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

const project = (root: string, writable = true): ArcProjectCandidate => ({
  descriptor: {
    format: 'arc-project',
    formatVersion: 2,
    guid: '00000000-0000-4000-8000-000000000001',
    name: 'Settings',
    engineVersion: '1.0.0',
    assetRoots: ['assets'],
    startupScenes: [],
    modules: [],
    paths: {
      source: 'Source',
      content: 'Content',
      config: 'Config',
      plugins: 'Plugins',
      saved: 'Saved',
      intermediate: 'Intermediate',
      build: 'Build',
    },
    plugins: [],
    defaultScene: null,
    targetPlatforms: [],
    toolchain: { compiler: 'auto', minimumVersion: '', generator: 'auto', architecture: 'x86_64', cppStandard: 20 },
    buildConfigurations: ['Debug', 'RelWithDebInfo', 'Shipping'],
    renderer: { backend: 'none', api: '', quality: 'standard' },
    cookProfiles: [],
    package: { applicationName: 'Settings', companyName: '', output: 'Build/Packages', regionChunks: true },
    settings: {
      editor: 'config/editor.settings.json',
      renderer: 'config/renderer.settings.json',
      input: 'config/input.settings.json',
    },
  },
  descriptorPath: path.join(root, 'Settings.arcproject'),
  projectRoot: root,
  compatibility: 'compatible',
  writable,
  diagnostics: [],
});

const testAiProviders = (): SettingsAiProviderService => {
  const credentials = new Map<AiProviderId, string>();
  const snapshot = (): AiProviderAccountsSnapshot => ({
    secureStorageAvailable: true,
    providers: [
      { id: 'openai', label: 'OpenAI', connected: credentials.has('openai') },
      { id: 'anthropic', label: 'Anthropic', connected: credentials.has('anthropic') },
    ],
  });
  return {
    snapshot,
    async connect(providerId, credential) {
      credentials.set(providerId, credential);
      return snapshot();
    },
    disconnect(providerId) {
      credentials.delete(providerId);
      return snapshot();
    },
    credential(providerId) {
      return credentials.get(providerId) ?? null;
    },
  };
};

describe('SettingsService', () => {
  it('layers project over user settings and resets the selected scope', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-settings-'));
    roots.push(root);
    const active = project(root);
    const service = new SettingsService(path.join(root, 'user.json'), () => active, testAiProviders());
    let snapshot = service.snapshot();

    snapshot = await service.update('user', { 'renderer.qualityTier': 'low' }, snapshot.revision);
    snapshot = await service.update('project', { 'renderer.qualityTier': 'high' }, snapshot.revision);
    expect(snapshot.values['renderer.qualityTier']).toBe('high');
    expect(snapshot.sources['renderer.qualityTier']).toBe('project');

    snapshot = await service.update('project', { 'renderer.qualityTier': undefined }, snapshot.revision);
    expect(snapshot.values['renderer.qualityTier']).toBe('low');
    expect(snapshot.sources['renderer.qualityTier']).toBe('user');
  });

  it('persists and validates the viewport grid color', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-settings-'));
    roots.push(root);
    const service = new SettingsService(path.join(root, 'user.json'), () => project(root), testAiProviders());
    let snapshot = service.snapshot();

    expect(snapshot.values['renderer.gridColor']).toBe('#33373D');
    snapshot = await service.update('user', { 'renderer.gridColor': '#4A5058' }, snapshot.revision);
    expect(snapshot.values['renderer.gridColor']).toBe('#4A5058');
    expect(snapshot.sources['renderer.gridColor']).toBe('user');
    const invalidGridColor = service.update('user', { 'renderer.gridColor': 'white' }, snapshot.revision);
    await expect(invalidGridColor).rejects.toThrow('#RRGGBB');
  });

  it('provides current OpenAI and Anthropic defaults while keeping credentials outside settings files', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-settings-'));
    roots.push(root);
    const userPath = path.join(root, 'user.json');
    const service = new SettingsService(userPath, () => project(root), testAiProviders());
    let snapshot = service.snapshot();

    expect(snapshot.values['ai.openai.model']).toBe('gpt-5.6-sol');
    expect(snapshot.values['ai.openai.reasoningEffort']).toBe('medium');
    expect(snapshot.values['ai.openai.storeResponses']).toBe(false);
    expect(snapshot.values['ai.anthropic.model']).toBe('claude-sonnet-5');
    expect(snapshot.values['ai.anthropic.effort']).toBe('high');
    expect(snapshot.values['ai.anthropic.maxOutputTokens']).toBe(16384);
    expect(snapshot.values['ai.openai.apiKey']).toBe('');
    expect(snapshot.values['ai.anthropic.apiKey']).toBe('');

    snapshot = await service.update('user', { 'ai.openai.apiKey': 'sk-project-secret' }, snapshot.revision);
    expect(snapshot.values['ai.openai.apiKey']).toBe('configured');
    expect(snapshot.sources['ai.openai.apiKey']).toBe('user');
    expect(service.providerCredential('openai')).toBe('sk-project-secret');

    snapshot = await service.update('user', { 'ai.anthropic.apiKey': 'sk-ant-secret' }, snapshot.revision);
    expect(snapshot.values['ai.anthropic.apiKey']).toBe('configured');
    expect(snapshot.sources['ai.anthropic.apiKey']).toBe('user');
    expect(service.providerCredential('anthropic')).toBe('sk-ant-secret');

    const projectUserSettings = path.join(root, 'Saved', 'Editor', 'settings.v1.json');
    const settingsText = fs.existsSync(projectUserSettings) ? fs.readFileSync(projectUserSettings, 'utf8') : '';
    expect(settingsText).not.toContain('sk-project-secret');
    expect(settingsText).not.toContain('sk-ant-secret');

    snapshot = await service.update('user', { 'ai.openai.apiKey': undefined }, snapshot.revision);
    expect(snapshot.values['ai.openai.apiKey']).toBe('');
    expect(snapshot.sources['ai.openai.apiKey']).toBe('default');
    expect(service.providerCredential('openai')).toBeNull();
  });

  it('rejects unknown, out-of-range, machine-only, stale, and read-only changes', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-settings-'));
    roots.push(root);
    let active = project(root);
    const service = new SettingsService(path.join(root, 'user.json'), () => active, testAiProviders());
    const revision = service.snapshot().revision;

    await expect(service.update('user', { unknown: true }, revision)).rejects.toThrow('Unknown setting');
    await expect(service.update('user', { 'renderer.targetFrameMilliseconds': Number.NaN }, revision)).rejects.toThrow(
      'finite',
    );
    await expect(service.update('project', { 'paths.externalShaderCompiler': 'tool' }, revision)).rejects.toThrow(
      'cannot be stored',
    );
    const updated = await service.update('user', { 'renderer.qualityTier': 'standard' }, revision);
    await expect(service.update('user', { 'renderer.qualityTier': 'low' }, revision)).rejects.toThrow('refresh');
    active = project(root, false);
    await expect(service.update('project', { 'renderer.qualityTier': 'low' }, updated.revision)).rejects.toThrow(
      'not writable',
    );
  });
});
