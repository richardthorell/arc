/// <reference types="vite/client" />

import type { ArcApi } from '../preload/preload';
import type { ArcAiProvidersApi } from '../preload/preloadWithAiProviders';

declare global {
  interface Window {
    arc: ArcApi;
    arcAiProviders: ArcAiProvidersApi;
  }
}
