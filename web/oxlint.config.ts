import { defineConfig } from "oxlint";
import core from "ultracite/oxlint/core";
import react from "ultracite/oxlint/react";
import shadcn from "ultracite/oxlint/shadcn";
import vitest from "ultracite/oxlint/vitest";

export default defineConfig({
  extends: [core, react, vitest, shadcn],
  ignorePatterns: [...(core.ignorePatterns ?? []), "src/components/ui/**"],
  jsPlugins: shadcn.jsPlugins,
});
