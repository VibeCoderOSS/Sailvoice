import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '127.0.0.1',
    watch: {
      ignored: ['**/runtime/**', '**/.venv/**', '**/.venv-align/**']
    }
  },
  build: {
    sourcemap: true,
    outDir: 'dist/renderer'
  }
});
