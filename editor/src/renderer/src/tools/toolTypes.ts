export type RenderGraphResource = {
  name: string;
  format: string;
  firstPass: number;
  lastPass: number;
  physicalResource: number;
  estimatedBytes: number;
};

export type RenderGraphTransition = {
  resource: string;
  before: string;
  after: string;
  beforePass: number;
  afterPass: number;
  beforeQueue: string;
  afterQueue: string;
};

export type EditorDiagnosticsSnapshot = {
  frameIndex: number;
  summary: string;
  passes: Array<{ name: string; milliseconds: number }>;
  timingsAvailable: boolean;
  graph: {
    executedPasses: string[];
    resources?: RenderGraphResource[];
    transitions?: RenderGraphTransition[];
    resourceCount: number;
    barrierCount: number;
    estimatedTransientBytes: number;
  };
  renderer: {
    path: string;
    renderScale: number;
    qualityTier: string;
    targetFrameMilliseconds: number;
    fallbackReasons: string[];
  };
  environment: {
    enabled: boolean;
    skyVisible: boolean;
    affectsLighting: boolean;
    source: string;
    qualityPath: string;
    atmosphereLutState: string;
    lightingState: string;
    cloudShadowResolution: number;
    fallback: string;
  };
  shadows: {
    cascades: number;
    directionalResolution: number;
    localAtlasResolution: number;
    localAllocations: number;
    localOccupiedTexels: number;
    localEvictions: number;
    localResolutionReductions: number;
    shadowedPointLights: number;
    shadowedSpotLights: number;
    staticCasters: number;
    dynamicCasters: number;
    localCacheHits: number;
    localCacheMisses: number;
    staticCacheHit: boolean;
    screenSpace: boolean;
    fallback: string;
  };
};
