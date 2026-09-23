import { app, ipcMain } from 'electron';
import path from 'node:path';

import type { AiProviderId } from '../common/aiProviderTypes';
import { AiProviderService } from './aiProviderService';
import './main';

void app.whenReady().then(() => {
  const providers = new AiProviderService(path.join(app.getPath('userData'), 'ai-provider-credentials.v1.json'));

  ipcMain.handle('ai-providers:snapshot', () => providers.snapshot());
  ipcMain.handle('ai-providers:connect', (_event, providerId: AiProviderId, credential: string) =>
    providers.connect(providerId, credential),
  );
  ipcMain.handle('ai-providers:disconnect', (_event, providerId: AiProviderId) => providers.disconnect(providerId));
});
