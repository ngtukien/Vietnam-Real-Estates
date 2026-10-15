import { defineConfig, devices } from "@playwright/test";

/**
 * `npm run e2e:real`: chỉ chạy `e2e/real.spec.ts` với backend THẬT đã chạy sẵn (không tự khởi động server).
 * Web: `VRE_WEB_URL` (mặc định vite preview 127.0.0.1:4173, proxy `/api` → 127.0.0.1:8000).
 * API phải chạy với `VRE_LLM_OFFLINE=1` + `LLM_PROVIDER=gemini` + `LLM_MODEL=gemini-flash-lite-latest` (khoá cache của eval).
 */
export default defineConfig({
  testDir: "e2e",
  testMatch: "real.spec.ts",
  timeout: 180_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: "list",
  use: { baseURL: process.env.VRE_WEB_URL ?? "http://127.0.0.1:4173", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
