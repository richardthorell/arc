import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { ArcProjectCandidate } from '../common/projectTypes';
import { SettingsService, type SettingsSecretCodec } from './settingsService';

const roots: string[] = [];

const testSecretCodec: SettingsSecretCodec = {
  encrypt: (value) => Buffer.from(`test:${value}`, 'utf8').toString('base64'),
  decrypt: (value) => Buffer.from(value, 'base64').toString('utf8').replace(/^test:/, ''),
};

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

describe('SettingsService', () => {
  it('layers project over user settings and resets the selected scope', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-settings-'));
    roots.push(root);
    const active = project(root);
    const service = new SettingsService(path.join(root, 'user.json'), () => active);
    let snapshot = service.snapshot();

    snapshot = service.update('user', { 'renderer.qualityTier': 'low' }, snapshot.revision);
    snapshot = service.update('project', { 'renderer.qualityTier': 'high' }, snapshot.revision);
    expect(snapshot.values['renderer.qualityTier']).toBe('high');
    expect(snapshot.sources['renderer.qualityTier']).toBe('project');

    snapshot = service.update('project', { 'renderer.qualityTier': undefined }, snapshot.revision);
    expect(snapshot.values['renderer.qualityTier']).toBe('low');
    expect(snapshot.sources['renderer.qualityTier']).toBe('user');
  });

  it('persists and validates the viewport grid color', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-settings-'));
    roots.push(root);
    const service = new SettingsService(path.join(root, 'user.json'), () => project(root));
    let snapshot = service.snapshot();

    expect(snapshot.values['renderer.gridColor']).toBe('#33373D');
    snapshot = service.update('user', { 'renderer.gridColor': '#4A5058' }, snapshot.revision);
    expect(snapshot.values['renderer.gridColor']).toBe('#4A5058');
    expect(snapshot.sources['renderer.gridColor']).toBe('user');
    expect(() => service.update('user', { 'renderer.gridColor': 'white' }, snapshot.revision)).toThrow('#RRGGBB');
  });

  it('provides OpenAI defaults and keeps API keys out of normal settings files', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-settings-'));
    roots.push(root);
    const userPath = path.join(root, 'user.json');
    const service = new SettingsService(userPath, () => project(root), testSecretCodec);
    let snapshot = service.snapshot();

    expect(snapshot.values['ai.openai.model']).toBe('gpt-6-sol');
    expect(snapshot.values['ai.openai.reasoningEffort']).toBe('medium');
    expect(snapshot.values['ai.openai.storeResponses']).toBe(false);
    expect(snapshot.values['ai.openai.apiKey']).toBe('');

    snapshot = service.update('user', { 'ai.openai.apiKey': 'sk-project-secret' }, snapshot.revision);
    expect(snapshot.values['ai.openai.apiKey']).toBe('configured');
    expect(snapshot.sources['ai.openai.apiKey']).toBe('user');
    expect(service.secretValue('ai.openai.apiKey')).toBe('sk-project-secret');

    const secretFile = fs.readFileSync(`${userPath}.secrets`, 'utf8');
    expect(secretFile).not.toContain('sk-project-secret');
    const projectUserSettings = path.join(root, 'Saved', 'Editor', 'settings.v1.json');
    expect(fs.existsSync(projectUserSettings) ? fs.readFileSync(projectUserSettings, 'utf8') : '').not.toContain(
      'sk-project-secret',
    );

    snapshot = service.update('user', { 'ai.openai.apiKey': undefined }, snapshot.revision);
    expect(snapshot.values['ai.openai.apiKey']).toBe('');
    expect(snapshot.sources['ai.openai.apiKey']).toBe('default');
    expect(service.secretValue('ai.openai.apiKey')).toBeNull();
  });

  it('rejects unknown, out-of-range, machine-only, stale, and read-only changes', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-settings-'));
    roots.push(root);
    let active = project(root);
    const service = new SettingsService(path.join(root, 'user.json'), () => active);
    const revision = service.snapshot().revision;

    expect(() => service.update('user', { unknown: true }, revision)).toThrow('Unknown setting');
    expect(() => service.update('user', { 'renderer.targetFrameMilliseconds': Number.NaN }, revision)).toThrow(
      'finite',
    );
    expect(() => service.update('project', { 'paths.externalShaderCompiler': 'tool' }, revision)).toThrow(
      'cannot be stored',
    );
    const updated = service.update('user', { 'renderer.qualityTier': 'standard' }, revision);
    expect(() => service.update('user', { 'renderer.qualityTier': 'low' }, revision)).toThrow('refresh');
    active = project(root, false);
    expect(() => service.update('project', { 'renderer.qualityTier': 'low' }, updated.revision)).toThrow(
      'not writable',
    );
  });
});
