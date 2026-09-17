import { fileURLToPath, URL } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The FastAPI viewer serves the build under /static, while a static host such as
// Cloudflare Pages serves it from the root. Both the base path and the output
// directory are overridable so one repository can build either target.
const staticHost = process.env.FLYGO_PUBLIC_BASE;
const staticOutDir = process.env.FLYGO_OUT_DIR;

export default defineConfig(({ command }) => ({
  base: staticHost ?? (command === "build" ? "/static/" : "/"),
  build: {
    emptyOutDir: true,
    outDir: staticOutDir ?? "../src/flygo/static",
  },
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("src", import.meta.url)),
    },
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  test: {
    environment: "node",
  },
}));
