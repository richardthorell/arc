import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

import type { AiProviderAccountsSnapshot, AiProviderConnectionStatus, AiProviderId } from '../common/aiProviderTypes';

type StoredCredentials = Partial<Record<AiProviderId, string>>;

type ProviderDefinition = {
  id: AiProviderId;
  label: string;
};

const providerDefinitions: readonly ProviderDefinition[] = [
  { id: 'openai', label: 'OpenAI' },
  { id: 'anthropic', label: 'Anthropic' },
] as const;

const providerById = new Map(providerDefinitions.map((provider) => [provider.id, provider] as const));

export type AiProviderSecureStorage = {
  isEncryptionAvailable(): boolean;
  getSelectedStorageBackend?(): string;
  encryptString(value: string): Buffer;
  decryptString(value: Buffer): string;
};

export type AiProviderCredentialValidator = (providerId: AiProviderId, credential: string) => Promise<void>;

type ElectronModule = {
  safeStorage?: AiProviderSecureStorage;
};

const electronSecureStorage = (): AiProviderSecureStorage => {
  // Forge's Vite main-process bundle is CommonJS, where import.meta.url is undefined.
  // Anchor createRequire to the emitted bundle filename instead so Electron's built-in
  // module can be resolved reliably in both development and packaged builds.
  const requireFromMain = createRequire(
    typeof __filename === 'string' ? __filename : path.join(process.cwd(), 'arc-electron-main.cjs'),
  );
  const electron = requireFromMain('electron') as ElectronModule;
  if (!electron.safeStorage) throw new Error('Electron secure credential storage is unavailable');
  return electron.safeStorage;
};

const validateProviderCredential: AiProviderCredentialValidator = async (providerId, credential) => {
  const signal = AbortSignal.timeout(10_000);
  const response =
    providerId === 'openai'
      ? await fetch('https://api.openai.com/v1/models', {
          headers: { Authorization: `Bearer ${credential}` },
          signal,
        })
      : await fetch('https://api.anthropic.com/v1/models', {
          headers: {
            'anthropic-version': '2023-06-01',
            'x-api-key': credential,
          },
          signal,
        });

  if (response.ok) return;
  const label = providerById.get(providerId)?.label ?? providerId;
  if (response.status === 401 || response.status === 403) throw new Error(`${label} rejected this API key`);
  throw new Error(`${label} credential validation failed (HTTP ${response.status})`);
};

export class AiProviderService {
  private readonly connectionStatus = new Map<
    AiProviderId,
    Exclude<AiProviderConnectionStatus, 'disconnected' | 'cold'>
  >();
  private readonly credentialGeneration = new Map<AiProviderId, number>();

  constructor(
    private readonly storagePath: string,
    private readonly secureStorage: () => AiProviderSecureStorage = electronSecureStorage,
    private readonly validateCredential: AiProviderCredentialValidator = validateProviderCredential,
  ) {}

  snapshot(): AiProviderAccountsSnapshot {
    const storage = this.secureStorageStatus();
    const stored = this.readStoredCredentials();
    return {
      secureStorageAvailable: storage.available,
      secureStorageDetail: storage.detail,
      providers: providerDefinitions.map((provider) => {
        const connected = Boolean(stored[provider.id]);
        return {
          id: provider.id,
          label: provider.label,
          connected,
          connectionStatus: connected ? (this.connectionStatus.get(provider.id) ?? 'cold') : 'disconnected',
        };
      }),
    };
  }

  async connect(providerId: AiProviderId, credential: string): Promise<AiProviderAccountsSnapshot> {
    const provider = providerById.get(providerId);
    if (!provider) throw new Error(`Unknown AI provider '${providerId}'`);
    const normalized = credential.trim();
    if (!normalized) throw new Error(`${provider.label} API key is required`);

    const storage = this.secureStorageStatus();
    if (!storage.available) throw new Error(storage.detail || 'Secure credential storage is unavailable');

    await this.validateCredential(providerId, normalized);
    const encrypted = this.secureStorage().encryptString(normalized).toString('base64');
    const stored = this.readStoredCredentials();
    stored[providerId] = encrypted;
    this.writeStoredCredentials(stored);
    this.bumpCredentialGeneration(providerId);
    this.connectionStatus.set(providerId, 'connected');
    return this.snapshot();
  }

  disconnect(providerId: AiProviderId): AiProviderAccountsSnapshot {
    if (!providerById.has(providerId)) throw new Error(`Unknown AI provider '${providerId}'`);
    const stored = this.readStoredCredentials();
    delete stored[providerId];
    this.writeStoredCredentials(stored);
    this.bumpCredentialGeneration(providerId);
    this.connectionStatus.delete(providerId);
    return this.snapshot();
  }

  async test(providerId: AiProviderId): Promise<AiProviderAccountsSnapshot> {
    if (!providerById.has(providerId)) throw new Error(`Unknown AI provider '${providerId}'`);
    const generation = this.credentialGeneration.get(providerId) ?? 0;
    try {
      const credential = this.credential(providerId);
      if (!credential) {
        if ((this.credentialGeneration.get(providerId) ?? 0) === generation) this.connectionStatus.delete(providerId);
        return this.snapshot();
      }
      await this.validateCredential(providerId, credential);
      if ((this.credentialGeneration.get(providerId) ?? 0) === generation)
        this.connectionStatus.set(providerId, 'connected');
    } catch {
      if ((this.credentialGeneration.get(providerId) ?? 0) === generation)
        this.connectionStatus.set(providerId, 'invalid');
    }
    return this.snapshot();
  }

  credential(providerId: AiProviderId): string | null {
    if (!providerById.has(providerId)) throw new Error(`Unknown AI provider '${providerId}'`);
    const encoded = this.readStoredCredentials()[providerId];
    if (!encoded) return null;
    const storage = this.secureStorageStatus();
    if (!storage.available) throw new Error(storage.detail || 'Secure credential storage is unavailable');
    return this.secureStorage().decryptString(Buffer.from(encoded, 'base64'));
  }

  private bumpCredentialGeneration(providerId: AiProviderId): void {
    this.credentialGeneration.set(providerId, (this.credentialGeneration.get(providerId) ?? 0) + 1);
  }

  private secureStorageStatus(): { available: boolean; detail?: string } {
    let storage: AiProviderSecureStorage;
    try {
      storage = this.secureStorage();
    } catch (error) {
      return { available: false, detail: error instanceof Error ? error.message : String(error) };
    }
    if (!storage.isEncryptionAvailable())
      return { available: false, detail: 'Operating-system credential encryption is unavailable' };
    if (process.platform === 'linux' && storage.getSelectedStorageBackend?.() === 'basic_text')
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
    const temporaryPath = `${this.storagePath}.tmp-${process.pid}`;
    fs.writeFileSync(temporaryPath, `${JSON.stringify(stored, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 });
    fs.renameSync(temporaryPath, this.storagePath);
    try {
      fs.chmodSync(this.storagePath, 0o600);
    } catch {
      // Windows does not provide POSIX file modes; safeStorage still protects the payload.
    }
  }
}
