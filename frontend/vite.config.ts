/// <reference types="vitest/config" />
import { fileURLToPath, URL } from 'node:url'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  // In development the page is on :5173 and the API on :8000. Proxying /api keeps the browser on a single origin,
  // exactly like production where FastAPI serves both, so the same code path is used everywhere.
  server: { port: 5173, strictPort: true, proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } } },
  preview: { port: 4173, strictPort: true, proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } } },
  build: {
    rolldownOptions: {
      output: {
        // Charts are the heaviest dependency; keeping them in their own file lets the browser cache them separately.
        codeSplitting: {
          groups: [
            { name: 'vendor-charts', test: /node_modules[/](recharts|d3-[^/]+|victory-vendor|decimal\.js-light|es-toolkit|internmap)[/]/ },
            { name: 'vendor-react', test: /node_modules[/](react|react-dom|react-router|react-router-dom|scheduler|@tanstack)[/]/ },
          ],
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
})
