import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vitest';

import type { AiProviderId } from '../common/aiProviderTypes';
import { AiProviderService, type AiProviderSecureStorage } from './aiProviderService';

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

const secureStorage = (available = true): AiProviderSecureStorage => ({
  isEncryptionAvailable: () => available,
  getSelectedStorageBackend: () => 'secretservice',
  encryptString: (value) => Buffer.from(`encrypted:${value}`, 'utf8'),
  decryptString: (value) => value.toString('utf8').replace(/^encrypted:/, ''),
});

describe('AiProviderService', () => {
  it('stores validated provider credentials securely without exposing them in snapshots', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-ai-providers-'));
    roots.push(root);
    const storagePath = path.join(root, 'providers.json');
    const validate = vi.fn(async (_providerId: AiProviderId, _credential: string) => undefined);
    const service = new AiProviderService(storagePath, () => secureStorage(), validate);

    expect(service.snapshot().providers).toEqual([
      { id: 'openai', label: 'OpenAI', connected: false },
      { id: 'anthropic', label: 'Anthropic', connected: false },
    ]);

    let snapshot = await service.connect('openai', 'sk-openai-secret');
    expect(validate).toHaveBeenCalledWith('openai', 'sk-openai-secret');
    expect(snapshot.providers.find((provider) => provider.id === 'openai')?.connected).toBe(true);
    expect(JSON.stringify(snapshot)).not.toContain('sk-openai-secret');
    expect(fs.readFileSync(storagePath, 'utf8')).not.toContain('sk-openai-secret');
    expect(service.credential('openai')).toBe('sk-openai-secret');

    await service.connect('anthropic', 'sk-ant-secret');
    expect(validate).toHaveBeenCalledWith('anthropic', 'sk-ant-secret');
    expect(service.credential('anthropic')).toBe('sk-ant-secret');

    await service.test('openai');
    expect(validate).toHaveBeenLastCalledWith('openai', 'sk-openai-secret');

    snapshot = service.disconnect('openai');
    expect(snapshot.providers.find((provider) => provider.id === 'openai')?.connected).toBe(false);
    expect(service.credential('openai')).toBeNull();
  });

  it('does not persist a credential when provider validation fails', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-ai-providers-'));
    roots.push(root);
    const storagePath = path.join(root, 'providers.json');
    const service = new AiProviderService(
      storagePath,
      () => secureStorage(),
      async () => {
        throw new Error('Provider rejected this API key');
      },
    );

    await expect(service.connect('openai', 'bad-key')).rejects.toThrow('Provider rejected this API key');
    expect(service.snapshot().providers.find((provider) => provider.id === 'openai')?.connected).toBe(false);
    expect(fs.existsSync(storagePath)).toBe(false);
  });

  it('refuses to save credentials when operating-system encryption is unavailable', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-ai-providers-'));
    roots.push(root);
    const service = new AiProviderService(
      path.join(root, 'providers.json'),
      () => secureStorage(false),
      async () => undefined,
    );

    expect(service.snapshot().secureStorageAvailable).toBe(false);
    await expect(service.connect('anthropic', 'secret')).rejects.toThrow('credential encryption is unavailable');
  });
});
