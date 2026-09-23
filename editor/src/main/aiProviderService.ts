import { safeStorage } from 'electron';
import fs from 'node:fs';
import path from 'node:path';

import type { AiProviderAccountsSnapshot, AiProviderId } from '../common/aiProviderTypes';

type StoredCredentials = Partial<Record<AiProviderId, string>>;

type ProviderDefinition = {
  id: AiProviderId;
  label: string;
  validate: (credential: string) => Promise<void>;
};

const providerDefinitions: readonly ProviderDefinition[] = [
  {
    id: 'openai',
    label: 'OpenAI',
    validate: async (credential) => {
      const response = await fetch('https://api.openai.com/v1/models', {
        headers: { Authorization: `Bearer ${credential}` },
        signal: AbortSignal.timeout(10_000),
      });
      if (response.ok) return;
      if (response.status === 401 || response.status === 403) throw new Error('OpenAI rejected this API key');
      throw new Error(`OpenAI credential validation failed (HTTP ${response.status})`);
    },
  },
  {
    id: 'anthropic',
    label: 'Anthropic',
    validate: async (credential) => {
      const response = await fetch('https://api.anthropic.com/v1/models', {
        headers: {
          'anthropic-version': '2023-06-01',
          'x-api-key': credential,
        },
        signal: AbortSignal.timeout(10_000),
      });
      if (response.ok) return;
      if (response.status === 401 || response.status === 403) throw new Error('Anthropic rejected this API key');
      throw new Error(`Anthropic credential validation failed (HTTP ${response.status})`);
    },
  },
] as const;

const providerById = new Map(providerDefinitions.map((provider) => [provider.id, provider] as const));

export class AiProviderService {
  constructor(private readonly storagePath: string) {}

  snapshot(): AiProviderAccountsSnapshot {
    const storage = this.secureStorageStatus();
    const stored = this.readStoredCredentials();
    return {
      secureStorageAvailable: storage.available,
      secureStorageDetail: storage.detail,
      providers: providerDefinitions.map((provider) => ({
        id: provider.id,
        label: provider.label,
        connected: Boolean(stored[provider.id]),
      })),
    };
  }

  async connect(providerId: AiProviderId, credential: string): Promise<AiProviderAccountsSnapshot> {
    const provider = providerById.get(providerId);
    if (!provider) throw new Error(`Unknown AI provider '${providerId}'`);
    const normalized = credential.trim();
    if (!normalized) throw new Error(`${provider.label} API key is required`);

    const storage = this.secureStorageStatus();
    if (!storage.available) throw new Error(storage.detail || 'Secure credential storage is unavailable');

    await provider.validate(normalized);
    const encrypted = safeStorage.encryptString(normalized).toString('base64');
    const stored = this.readStoredCredentials();
    stored[providerId] = encrypted;
    this.writeStoredCredentials(stored);
    return this.snapshot();
  }

  disconnect(providerId: AiProviderId): AiProviderAccountsSnapshot {
    if (!providerById.has(providerId)) throw new Error(`Unknown AI provider '${providerId}'`);
    const stored = this.readStoredCredentials();
    delete stored[providerId];
    this.writeStoredCredentials(stored);
    return this.snapshot();
  }

  credential(providerId: AiProviderId): string | null {
    const encoded = this.readStoredCredentials()[providerId];
    if (!encoded) return null;
    const storage = this.secureStorageStatus();
    if (!storage.available) throw new Error(storage.detail || 'Secure credential storage is unavailable');
    return safeStorage.decryptString(Buffer.from(encoded, 'base64'));
  }

  private secureStorageStatus(): { available: boolean; detail?: string } {
    if (!safeStorage.isEncryptionAvailable())
      return { available: false, detail: 'Operating-system credential encryption is unavailable' };
    if (process.platform === 'linux' && safeStorage.getSelectedStorageBackend() === 'basic_text')
      return {
        available: false,
        detail: 'A Linux secret store is required before ARC can save AI provider credentials securely',
      };
    return { available: true };
  }

  private readStoredCredentials(): StoredCredentials {
    try {
      if (!fs.existsSync(this.storagePath)) return {};
      const parsed = JSON.parse(fs.readFileSync(this.storagePath, 'utf8')) as unknown;
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {};
      const record = parsed as Record<string, unknown>;
      const stored: StoredCredentials = {};
      for (const provider of providerDefinitions) {
        const value = record[provider.id];
        if (typeof value === 'string' && value) stored[provider.id] = value;
      }
      return stored;
    } catch {
      return {};
    }
  }

  private writeStoredCredentials(stored: StoredCredentials): void {
    fs.mkdirSync(path.dirname(this.storagePath), { recursive: true });
    const temporaryPath = `${this.storagePath}.tmp`;
    fs.writeFileSync(temporaryPath, `${JSON.stringify(stored, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 });
    fs.renameSync(temporaryPath, this.storagePath);
  }
}
