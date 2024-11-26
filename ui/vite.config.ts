import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          // Create vendor chunks for node_modules
          if (id.includes('node_modules')) {
            return 'vendor';
          }

          // Handle JSON data files
          if (id.includes('/data/')) {
            const fileName = path.basename(id);

            // Skip non-compressed files in production
            if (mode === 'production' && !fileName.includes('.compressed.json')) {
              return null;
            }

            // Group by file size
            const size = fileName.length; // Simple size estimation
            const prefix = mode === 'production' ? 'compressed' : 'regular';

            if (size > 500000) {
              return `${prefix}-data-large-${fileName}`;
            }
            if (size > 100000) {
              return `${prefix}-data-medium-${fileName}`;
            }
            return `${prefix}-data-small-${fileName}`;
          }
        },
        entryFileNames: 'entries/[name]-[hash].js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]'
      },
      input: {
        main: path.resolve(__dirname, 'index.html')
      }
    },
    chunkSizeWarningLimit: 3000,
    reportCompressedSize: false,
    sourcemap: mode !== 'production',
    minify: mode === 'production' ? 'esbuild' : false,
    target: 'esnext',
    assetsInlineLimit: 4096,
  },
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
    ]
  },
  server: {
    fs: {
      strict: false
    }
  },
  esbuild: {
    target: 'esnext',
    legalComments: 'none',
    treeShaking: true,
  }
}));