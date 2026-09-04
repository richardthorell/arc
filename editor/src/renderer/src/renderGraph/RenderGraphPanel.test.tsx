// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { RenderGraphPanel } from './RenderGraphPanel';

const diagnostics = {
  frameIndex: 42,
  summary: 'Frame 42',
  passes: [
    { name: 'depth_prepass', milliseconds: 0.4 },
    { name: 'deferred_lighting', milliseconds: 1.2 },
  ],
  timingsAvailable: true,
  graph: {
    executedPasses: ['depth_prepass', 'deferred_lighting'],
    resources: [
      {
        name: 'scene_depth',
        format: 'd32_float',
        firstPass: 0,
        lastPass: 1,
        physicalResource: 2,
        estimatedBytes: 4 * 1024 * 1024,
      },
    ],
    transitions: [
      {
        resource: 'scene_depth',
        before: 'depth_attachment',
        after: 'shader_read',
        beforePass: 0,
        afterPass: 1,
        beforeQueue: 'graphics',
        afterQueue: 'graphics',
      },
    ],
    resourceCount: 1,
    barrierCount: 1,
    estimatedTransientBytes: 4 * 1024 * 1024,
  },
  renderer: {
    path: 'deferred',
    renderScale: 1,
    qualityTier: 'standard',
    targetFrameMilliseconds: 16.67,
    fallbackReasons: [],
  },
  environment: {
    enabled: true,
    skyVisible: true,
    affectsLighting: true,
    source: 'physical',
    qualityPath: 'analytic',
    atmosphereLutState: 'fallback',
    lightingState: 'diffuse',
    cloudShadowResolution: 0,
    fallback: '',
  },
  shadows: {
    cascades: 4,
    directionalResolution: 2048,
    localAtlasResolution: 4096,
    localAllocations: 0,
    localOccupiedTexels: 0,
    localEvictions: 0,
    localResolutionReductions: 0,
    shadowedPointLights: 0,
    shadowedSpotLights: 0,
    staticCasters: 0,
    dynamicCasters: 0,
    localCacheHits: 0,
    localCacheMisses: 0,
    staticCacheHit: false,
    screenSpaceShadows: false,
    virtualShadowMaps: false,
    virtualAddressSpaces: 0,
    virtualPageCapacity: 0,
    virtualResidentPages: 0,
    virtualDirtyPages: 0,
    virtualRenderedPages: 0,
    virtualReusedPages: 0,
    virtualEvictions: 0,
    virtualParentFallbacks: 0,
    virtualFailedRequests: 0,
    virtualMemoryBytes: 0,
    fallback: '',
  },
  textureStreaming: {
    gpuBudgetBytes: 512 * 1024 * 1024,
    gpuResidentBytes: 96 * 1024 * 1024,
    cpuBudgetBytes: 128 * 1024 * 1024,
    cpuCachedBytes: 12 * 1024 * 1024,
    uploadBudgetBytes: 64 * 1024 * 1024,
    uploadedBytes: 4 * 1024 * 1024,
    ioReadBytes: 6 * 1024 * 1024,
    ioFailedBytes: 0,
    ioCompletedReads: 18,
    ioFailedReads: 0,
    ioInFlightReads: 3,
    ioLatencyMilliseconds: 4.5,
    uploadLatencyMilliseconds: 1.2,
    streamedTextures: 4,
    virtualTextures: 1,
    residentMips: 19,
    residentPages: 72,
    requestedMips: 2,
    requestedPages: 5,
    failedMips: 0,
    failedPages: 1,
    evictions: 3,
    feedbackOverflow: 0,
    parentFallbacks: 2,
    cacheHitRate: 0.875,
    overBudget: false,
    fallback: '',
    resources: [],
  },
  gpuScene: {
    enabled: true,
    hzbOcclusion: true,
    historyValid: true,
    capacity: 256,
    activeInstances: 12,
    uploadedInstances: 2,
    uploadedRanges: 2,
    uploadedBytes: 640,
    geometryTableEntries: 4,
    materialTableEntries: 3,
    textureTableEntries: 7,
    samplerTableEntries: 1,
    skinPaletteTableEntries: 0,
    sharedVertexHeapBytes: 4096,
    sharedIndexHeapBytes: 1024,
    candidateInstances: 12,
    visibleInstances: 9,
    frustumRejected: 2,
    distanceRejected: 1,
    occlusionRejected: 0,
    activePipelineBins: 4,
    indirectCommands: 4,
    overflowRecords: 0,
    cpuSubmissions: 0,
    fallback: '',
  },
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('RenderGraphPanel', () => {
  it('renders executed passes, resource lifetimes, and barriers in the graph canvas', async () => {
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        host: {
          query: vi.fn().mockResolvedValue({ succeeded: true, payload: diagnostics }),
        },
      },
    });

    render(<RenderGraphPanel />);
    await waitFor(() => expect(screen.getByRole('img', { name: 'Executed graph for frame 42' })).toBeInTheDocument());
    expect(screen.getAllByText('depth_prepass').length).toBeGreaterThan(0);
    expect(screen.getAllByText(/scene_depth/).length).toBeGreaterThan(0);
    expect(document.querySelector('.render-graph-transition')).toBeInTheDocument();
    expect(screen.getByLabelText('Texture streaming diagnostics')).toHaveTextContent('Hit 87.5%');
    expect(screen.getByLabelText('GPU Scene resource tables')).toHaveTextContent('4 geometry');

    await userEvent.type(screen.getByLabelText('Filter render graph'), 'scene_depth');
    expect(screen.getByRole('img', { name: 'Executed graph for frame 42' })).toBeInTheDocument();
  });
});
