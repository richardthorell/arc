import { builtinModules } from 'node:module';
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: '.vite/build',
    emptyOutDir: false,
    lib: {
      entry: 'src/main/arcMcpStdio.ts',
      formats: ['es'],
      fileName: () => 'arc-mcp-stdio.js',
    },
    rollupOptions: {
      external: [...builtinModules, ...builtinModules.map((name) => `node:${name}`)],
    },
  },
});
