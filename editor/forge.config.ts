import type { ForgeConfig } from '@electron-forge/shared-types';
import { MakerDeb } from '@electron-forge/maker-deb';
import { MakerRpm } from '@electron-forge/maker-rpm';
import { MakerSquirrel } from '@electron-forge/maker-squirrel';
import { MakerZIP } from '@electron-forge/maker-zip';
import { AutoUnpackNativesPlugin } from '@electron-forge/plugin-auto-unpack-natives';
import { VitePlugin } from '@electron-forge/plugin-vite';

const bundledHost = process.env.ARC_PACKAGED_HOST_PATH;
const bundledProjectTool = process.env.ARC_PACKAGED_PROJECT_TOOL_PATH;
const bundledTemplates = process.env.ARC_PACKAGED_TEMPLATES_PATH;
const bundledRenderVulkan = process.env.ARC_PACKAGED_RENDER_VULKAN_PATH;
if (process.env.ARC_REQUIRE_PACKAGED_HOST === '1' && !bundledHost) {
  throw new Error('ARC_PACKAGED_HOST_PATH is required for packaged editor builds');
}

const config: ForgeConfig = {
  packagerConfig: {
    asar: true,
    executableName: 'arc-editor',
    extraResource: [bundledHost, bundledProjectTool, bundledTemplates, bundledRenderVulkan].filter(
      (entry): entry is string => Boolean(entry),
    ),
  },
  rebuildConfig: {},
  makers: [new MakerSquirrel({}), new MakerZIP({}, ['darwin']), new MakerDeb({}), new MakerRpm({})],
  plugins: [
    new AutoUnpackNativesPlugin({}),
    new VitePlugin({
      build: [
        {
          entry: 'src/main/main.ts',
          config: 'vite.main.config.ts',
          target: 'main',
        },
        {
          entry: 'src/preload/preload.ts',
          config: 'vite.preload.config.ts',
          target: 'preload',
        },
        {
          entry: 'src/main/arcMcpStdio.ts',
          config: 'vite.mcp.config.ts',
          target: 'main',
        },
      ],
      renderer: [
        {
          name: 'main_window',
          config: 'vite.renderer.config.ts',
        },
      ],
    }),
  ],
};

export default config;
