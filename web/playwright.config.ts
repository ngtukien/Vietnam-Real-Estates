import path from "node:path";
import { defineConfig, devices } from "@playwright/test";

/**
 * E2E chạy với API ở chế độ fake (`VRE_FAKE_BACKENDS=1`, api-contract §6) trên 127.0.0.1:8000.
 * API đã chạy sẵn (CI khởi động trước) ⇒ dùng lại; nếu không, tự khởi động uvicorn từ `VRE_API_DIR`
 * (mặc định gốc repo) bằng `VRE_PY` (mặc định `python`). Một worker, chỉ chromium.
 */
const apiDir = process.env.VRE_API_DIR ?? path.resolve(import.meta.dirname, "..");
const python = process.env.VRE_PY ?? "python";

export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: { baseURL: "http://127.0.0.1:5173", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: `"${python}" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --timeout-keep-alive 5`,
      cwd: apiDir,
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: true,
      env: { VRE_FAKE_BACKENDS: "1", VRE_LOAD_DOTENV: "0", PYTHONUTF8: "1", VRE_CHAT_RATE_PER_MIN: "1000" },
      timeout: 120_000,
    },
    {
      command: "npm run dev",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
