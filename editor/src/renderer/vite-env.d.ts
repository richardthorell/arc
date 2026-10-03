/// <reference types="vite/client" />

import type { ArcApi } from '../preload/preload';
import type { ArcAiRuntimeApi } from '../preload/aiRuntimeBridge';

declare global {
  interface Window {
    arc: ArcApi;
    arcAiRuntime: ArcAiRuntimeApi;
  }
}
