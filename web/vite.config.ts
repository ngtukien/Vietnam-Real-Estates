import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const apiTarget = process.env.VRE_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    // SSE đi qua proxy: không nén, không buffer.
    proxy: { "/api": { target: apiTarget, changeOrigin: false } },
  },
  preview: { host: "127.0.0.1", port: 4173, proxy: { "/api": { target: apiTarget } } },
  build: {
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("maplibre-gl")) return "maplibre";
          if (id.includes("recharts") || id.includes("d3-")) return "charts";
          if (id.includes("force-graph") || id.includes("kapsule")) return "graph";
          return undefined;
        },
      },
    },
  },
  test: { include: ["src/**/*.test.ts"], environment: "node" },
});
