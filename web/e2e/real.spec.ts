import { expect, test } from "@playwright/test";

/**
 * Kịch bản thủ công `npm run e2e:real`: webapp chạy với backend THẬT (Qdrant + Neo4j + mô hình định giá),
 * API khởi động với `VRE_LLM_OFFLINE=1` và đúng provider/model của lượt eval (xem `results/eval/raw/run_meta.json`)
 * ⇒ LLM chỉ trả từ cache đĩa, không gọi mạng. Vì vậy chỉ dùng câu hỏi đã có câu trả lời trong
 * `results/eval/raw/answers.jsonl`: AG01 (aggregate, route graph) và LK01 (lookup, route basic_rag).
 * Câu chưa có trong cache sẽ hiện lỗi LLMCacheMiss — đó là lỗi đúng, không phải lỗi test.
 * Không khẳng định văn bản cụ thể của câu trả lời (đổi model sinh ⇒ đổi câu chữ); chỉ khẳng định cấu trúc.
 */
const AG01 = "Giá trung vị mỗi m2 căn hộ chung cư ở Cầu Giấy là bao nhiêu?";
const LK01 = "Mình thấy tin bán căn hộ 3 phòng ngủ 80m2 ở Sun Feliza Suites Cầu Giấy giá 8 tỷ, căn đó có mấy WC vậy?";
const LK01_LISTING = "3092923782945354";

// Lần đầu nạp bộ mã hoá truy vấn + mô hình mất ~20 s.
test.setTimeout(180_000);
test.use({ actionTimeout: 15_000 });
const SLOW = { timeout: 150_000 };

async function ask(page: import("@playwright/test").Page, question: string): Promise<void> {
  await page.goto("/");
  await page.getByLabel("Câu hỏi").fill(question);
  await page.getByRole("button", { name: "Gửi" }).click();
}

test("thật — chat hybrid, câu aggregate: route graph + Cypher + câu trả lời từ cache", async ({ page }) => {
  await ask(page, AG01);
  const answer = page.getByTestId("answer");
  await expect(page.getByTestId("usage")).toBeVisible(SLOW);
  await expect(answer).not.toBeEmpty();
  await expect(page.getByTestId("route-badge")).toContainText("Route:");
  await expect(page.getByTestId("cypher-panel")).toBeVisible();
  await expect(page.getByTestId("cypher-panel")).toContainText("Cầu Giấy");
  await expect(page.getByTestId("usage")).toContainText("từ cache");
  await expect(page.getByTestId("run-error")).toHaveCount(0);
});

test("thật — chat hybrid, câu lookup: route basic_rag + nguồn + trích dẫn tin", async ({ page }) => {
  await ask(page, LK01);
  const answer = page.getByTestId("answer");
  await expect(page.getByTestId("usage")).toBeVisible(SLOW);
  await expect(answer).toContainText("Tin#");
  await expect(page.getByTestId("route-badge")).toContainText("Route:");
  await expect(page.getByTestId("source-card").first()).toBeVisible();
  // Tin đúng của LK01 (ground truth) phải nằm trong các nguồn đã truy hồi.
  await expect(page.getByTestId("source-card").filter({ hasText: LK01_LISTING })).not.toHaveCount(0);
  await expect(page.getByTestId("usage")).toContainText("từ cache");
  await expect(page.getByTestId("run-error")).toHaveCount(0);
});

test("thật — so sánh 4 hệ trên AG01: 4 cột, cả 4 trả lời từ cache", async ({ page }) => {
  await page.goto("/compare");
  await page.getByLabel("Câu hỏi").fill(AG01);
  await page.getByRole("button", { name: "So sánh" }).click();
  await expect(page.getByTestId("compare-done")).toBeVisible(SLOW);
  for (const s of ["llm_only", "basic_rag", "graph", "hybrid"]) {
    const col = page.getByTestId(`compare-col-${s}`);
    await expect(col).toBeVisible();
    await expect(col.getByTestId("answer")).not.toBeEmpty();
    await expect(col.getByTestId("run-error")).toHaveCount(0);
  }
});

test("thật — định giá (không dùng LLM): kết quả + nhận định giá chào", async ({ page }) => {
  await page.goto("/valuate");
  await page.getByLabel("Tỉnh/thành *").selectOption("Hồ Chí Minh");
  await page.getByLabel("Quận/huyện *").selectOption("Quận 7");
  await page.getByLabel("Diện tích (m²) *").fill("70");
  await page.getByLabel(/Giá chào/).fill("3");
  await page.getByRole("button", { name: "Định giá" }).click();
  await expect(page.getByTestId("valuation-result")).toBeVisible(SLOW);
  await expect(page.getByTestId("verdict")).toBeVisible();
});
