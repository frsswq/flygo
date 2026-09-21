import { defineConfig } from "vitest/config";

// The browser regression drives a real Chromium, so it has its own config and
// runs through `npm run test:browser` instead of the default unit-test run.
export default defineConfig({
  test: {
    environment: "node",
    include: ["browser/**/*.test.ts"],
    testTimeout: 120_000,
  },
});
