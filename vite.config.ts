import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: './',
  // Pre-bundle up front so the dev server never force-reloads the page mid-session.
  optimizeDeps: { include: ['lil-gui'] },
  build: {
    target: 'es2022',
    sourcemap: true,
  },
  test: {
    include: ['tests/**/*.test.ts'],
  },
});
