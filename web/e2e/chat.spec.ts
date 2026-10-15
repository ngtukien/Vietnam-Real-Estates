import { expect, test } from "@playwright/test";
import { expectNoSeriousA11y, sseBody } from "./helpers";

test("chat hybrid: stream câu trả lời, nguồn, Cypher, route, usage", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Câu hỏi").fill("căn hộ 2pn q7 dưới 4 tỷ");
  await page.getByRole("button", { name: "Gửi" }).click();
  const answer = page.getByTestId("answer");
  await expect(answer).toContainText("Tin#");
  await expect(page.getByTestId("source-card").first()).toBeVisible();
  await expect(page.getByTestId("cypher-panel")).toBeVisible();
  await expect(page.getByTestId("route-badge")).toBeVisible();
  await expect(page.getByTestId("usage")).toBeVisible();
  const link = answer.getByRole("link").first();
  const href = await link.getAttribute("href");
  await link.click();
  await expect(page.locator(href as string)).toBeFocused();
  await expect(page.getByRole("button", { name: /căn hộ 2pn q7/ })).toBeVisible();
  await expectNoSeriousA11y(page);
});

test("chat: route định giá của hybrid", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Câu hỏi").fill("định giá căn hộ 70m2 quận 7");
  await page.getByRole("button", { name: "Gửi" }).click();
  await expect(page.getByTestId("valuation-result")).toBeVisible();
});

test("chat: lỗi trong stream hiển thị rõ", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Câu hỏi").fill("[fake:error] hỏi gì đó");
  await page.getByRole("button", { name: "Gửi" }).click();
  await expect(page.getByTestId("run-error")).toContainText("LLM");
});

test("chat: Cypher lỗi được hiển thị (không bị ẩn)", async ({ page }) => {
  const base = { request_id: "deadbeef", system: "graph" };
  await page.route("**/api/chat", (route) =>
    route.fulfill({
      status: 200,
      headers: { "content-type": "text/event-stream" },
      body: sseBody([
        ["meta", { ...base, question: "q", started_at: "2026-10-15T07:14:00Z" }],
        [
          "cypher",
          { ...base, cypher: "MATCH (n:Foo) RETURN n LIMIT 50", columns: [], rows: [], error: "Neo.ClientError.Statement.SyntaxError: Invalid input", attempts: 3 },
        ],
        ["sources", { ...base, sources: [] }],
        ["answer", { ...base, answer: "Không tìm được dữ liệu.", cited: [], unsupported_citations: [], fallback: false, route: null }],
        ["usage", { ...base, latency_ms: 900, input_tokens: 10, output_tokens: 5, usd: 0.0001, cached: false }],
        ["done", { ...base, status: "ok" }],
      ]),
    }),
  );
  await page.goto("/");
  await page.getByLabel("Câu hỏi").fill("quận nào đắt nhất");
  await page.getByLabel("GraphRAG (Cypher)").check();
  await page.getByRole("button", { name: "Gửi" }).click();
  await expect(page.getByTestId("cypher-error")).toContainText("SyntaxError");
});

test("?q= chỉ điền sẵn, không tự gửi; không có token trên URL", async ({ page }) => {
  let posted = false;
  page.on("request", (r) => {
    if (r.url().includes("/api/chat")) posted = true;
  });
  await page.goto("/?q=nh%C3%A0%20Q1");
  await expect(page.getByLabel("Câu hỏi")).toHaveValue("nhà Q1");
  await page.waitForTimeout(500);
  expect(posted).toBe(false);
  expect(page.url()).not.toMatch(/key|token/i);
});
