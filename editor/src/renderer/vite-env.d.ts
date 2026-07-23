/// <reference types="vite/client" />

import type { ArcApi } from '../preload/preload';

declare global {
  interface Window {
    arc: ArcApi;
  }
}
