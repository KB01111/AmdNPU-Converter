import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/ui",
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: { baseURL: "http://127.0.0.1:1430", channel: process.env.CI ? undefined : "msedge", trace: "retain-on-failure" },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 1430",
    url: "http://127.0.0.1:1430",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
