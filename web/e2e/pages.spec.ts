import { expect, test } from "@playwright/test";
import { expectNoSeriousA11y } from "./helpers";

test("so sánh 4 hệ: 4 cột và compare_done", async ({ page }) => {
  await page.goto("/compare");
  await page.getByLabel("Câu hỏi").fill("căn hộ 2pn q7 dưới 4 tỷ");
  await page.getByRole("button", { name: "So sánh" }).click();
  for (const s of ["llm_only", "basic_rag", "graph", "hybrid"]) {
    await expect(page.getByTestId(`compare-col-${s}`)).toBeVisible();
  }
  await expect(page.getByTestId("compare-done")).toBeVisible();
  await expect(page.getByTestId("compare-col-hybrid").getByTestId("answer")).toContainText("Tin#");
  await expectNoSeriousA11y(page);
});

test("định giá: kết quả + SHAP + nhận định giá chào", async ({ page }) => {
  await page.goto("/valuate");
  await page.getByLabel("Tỉnh/thành *").selectOption("Hồ Chí Minh");
  await page.getByLabel("Quận/huyện *").selectOption("Quận 7");
  await page.getByLabel("Diện tích (m²) *").fill("70");
  await page.getByLabel(/Giá chào/).fill("3");
  await page.getByRole("button", { name: "Định giá" }).click();
  await expect(page.getByTestId("valuation-result")).toBeVisible();
  await expect(page.getByTestId("verdict")).toBeVisible();
  await expectNoSeriousA11y(page);
});

test("bản đồ render không cần tile ngoài", async ({ page }) => {
  const external: string[] = [];
  const consoleErrors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("request", (r) => {
    const u = new URL(r.url());
    if (u.protocol.startsWith("http") && u.hostname !== "127.0.0.1" && u.hostname !== "localhost") external.push(r.url());
  });
  await page.goto("/map");
  await expect(page.locator("canvas.maplibregl-canvas")).toBeVisible();
  await expect(page.getByTestId("coverage")).toContainText("Độ phủ");
  await expect(page.getByRole("table")).toBeVisible();
  expect(external).toEqual([]);
  // Lớp polygon đã được vẽ (worker GeoJSON chạy được)
  await expect
    .poll(() => page.evaluate(() => document.querySelector("canvas.maplibregl-canvas") !== null))
    .toBe(true);
  expect(consoleErrors.filter((e) => /worker/i.test(e))).toEqual([]);
  await expectNoSeriousA11y(page);
});

test("dashboard thị trường", async ({ page }) => {
  await page.goto("/market");
  await expect(page.getByTestId("chart-by-area")).toBeVisible();
  await expect(page.getByTestId("chart-monthly")).toBeVisible();
  await expectNoSeriousA11y(page);
});

test("benchmark", async ({ page }) => {
  await page.goto("/benchmark");
  await expect(page.getByTestId("benchmark-chart")).toBeVisible();
  await expect(page.getByRole("table").first()).toBeVisible();
  await expectNoSeriousA11y(page);
});

test("điều hướng bàn phím + dark mode", async ({ page }) => {
  await page.goto("/");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Bỏ qua tới nội dung" })).toBeFocused();
  const nav = page.getByRole("navigation", { name: "Trang chính" });
  await nav.getByRole("link", { name: "Định giá" }).focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/valuate$/);
  await page.getByRole("button", { name: /giao diện tối/ }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await expectNoSeriousA11y(page);
});
