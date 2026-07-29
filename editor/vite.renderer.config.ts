import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const editorRoot = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: 'src/renderer',
  plugins: [react()],
  build: {
    // Electron Forge supplies the renderer name but Vite otherwise resolves
    // its relative output directory against `root`, outside the packaged ASAR.
    outDir: path.join(editorRoot, '.vite', 'renderer', 'main_window'),
  },
});
