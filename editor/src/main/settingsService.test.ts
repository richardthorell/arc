import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { ArcProjectCandidate } from '../common/projectTypes';
import { SettingsService } from './settingsService';

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

const project = (root: string, writable = true): ArcProjectCandidate => ({
  descriptor: {
    format: 'arc-project',
    formatVersion: 1,
    guid: '00000000-0000-4000-8000-000000000001',
    name: 'Settings',
    engineVersion: '1.0.0',
    assetRoots: ['assets'],
    startupScenes: [],
    modules: [],
    extensions: [],
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
