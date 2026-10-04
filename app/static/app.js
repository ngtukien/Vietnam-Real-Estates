const $ = (id) => document.getElementById(id);
const messages = $("messages");
const form = $("chat-form");
const queryInput = $("query");
const sendButton = $("send");
const ROUTES = { basic_rag: "Basic RAG", graph: "Graph", hybrid: "Graph + RAG" };
const FILTER_NAMES = {
  province: "tỉnh", district: "quận", property_type: "loại hình", bedrooms: "phòng ngủ",
  min_price_mil: "giá từ (triệu)", max_price_mil: "giá đến (triệu)", min_area: "từ (m²)", max_area: "đến (m²)",
};
let turn = 0;

async function loadInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    for (const [key, system] of Object.entries(info.systems)) {
      const option = new Option(system.name + (system.available ? "" : " (cần Neo4j)"), key);
      option.disabled = !system.available;
      $("system").add(option);
    }
    $("system").value = info.systems.hybrid.available ? "hybrid" : "basic_rag";
    const n = (value) => Number(value).toLocaleString("vi-VN");
    const scope = info.stats.index === "full" ? "toàn bộ dataset" : "mẫu";
    $("stats").textContent = `${n(info.stats.listings)} tin (${scope}) · ${n(info.stats.chunks)} chunk`
      + (info.graph_error ? ` · ${info.graph_error}` : " · đồ thị sẵn sàng");
  } catch {
    $("stats").textContent = "Không kết nối được server";
  }
}

function bubble(className, text) {
  const element = document.createElement("div");
  element.className = `bubble ${className}`;
  if (text) element.textContent = text;
  messages.append(element);
  element.scrollIntoView({ behavior: "smooth", block: "end" });
  return element;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Câu trả lời dạng văn bản; mỗi [Tin#ID] thành liên kết tới thẻ tin nguồn bên dưới.
function answerText(answer, prefix) {
  const paragraph = element("div", "answer");
  const known = new Set(answer.sources.map((s) => s.id));
  for (const part of answer.answer.split(/(\[Tin#\d+\])/)) {
    const match = part.match(/^\[Tin#(\d+)\]$/);
    if (match && known.has(Number(match[1]))) {
      const link = element("a", "cite-link", `#${match[1]}`);
      link.href = `#${prefix}-${match[1]}`;
      paragraph.append(link);
    } else {
      paragraph.append(document.createTextNode(part));
    }
  }
  return paragraph;
}

function howFound(answer) {
  const parts = [answer.system_name];
  if (answer.route) parts.push(`router chọn ${ROUTES[answer.route] || answer.route}`);
  if (answer.fallback) parts.push("đồ thị không có kết quả, đã dùng RAG");
  const filters = Object.entries(answer.filters).map(([k, v]) => `${FILTER_NAMES[k] || k}: ${v}`);
  if (filters.length) parts.push(`bộ lọc ${filters.join(", ")}`);
  if (answer.note) parts.push(answer.note);
  parts.push(`${answer.elapsed_ms} ms · ${answer.tokens.toLocaleString("vi-VN")} token`);
  return parts.join(" · ");
}

function sourceCard(source, prefix, cited) {
  const card = $("source-card").content.firstElementChild.cloneNode(true);
  card.id = `${prefix}-${source.id}`;
  const set = (selector, text) => { card.querySelector(selector).textContent = text; };
  set(".type", source.property_type);
  set(".cite", `Tin #${source.id}${cited ? " · được trích" : ""}${source.published_at ? " · đăng " + source.published_at : ""}`);
  set(".title", source.title);
  set(".price", source.price);
  set(".area", source.area);
  const rooms = [source.bedrooms && `${source.bedrooms} PN`, source.bathrooms && `${source.bathrooms} WC`].filter(Boolean);
  if (rooms.length) set(".rooms", rooms.join(" · "));
  else card.querySelector(".rooms-cell").remove();
  set(".address", [source.project, source.address].filter(Boolean).join(" — ") || "Chưa rõ địa chỉ");
  set(".description", source.description || "Không có mô tả");
  return card;
}

function renderAnswer(answer) {
  const prefix = `t${++turn}`;
  const container = bubble("bot");
  container.append(answerText(answer, prefix));
  container.append(element("p", "note muted", howFound(answer)));
  if (answer.cypher) {
    const details = element("details", "evidence");
    details.append(element("summary", "", "Cypher đã chạy"), element("pre", "", answer.cypher));
    container.append(details);
  }
  if (answer.sources.length) {
    const cited = new Set(answer.cited);
    const list = element("div", "sources");
    for (const source of answer.sources) list.append(sourceCard(source, prefix, cited.has(source.id)));
    container.append(element("p", "sources-title muted", "Tin nguồn"), list);
  }
  container.scrollIntoView({ behavior: "smooth", block: "end" });
}

async function ask(query) {
  $("welcome")?.remove();
  bubble("user", query);
  const typing = bubble("typing", "Đang tìm và soạn câu trả lời…");
  sendButton.disabled = true;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, system: $("system").value }),
    });
    const answer = await response.json();
    typing.remove();
    if (answer.error && !answer.answer) bubble("error", answer.error);
    else renderAnswer(answer);
  } catch {
    typing.remove();
    bubble("error", "Không kết nối được server. Kiểm tra app/server.py còn chạy.");
  } finally {
    sendButton.disabled = false;
    queryInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  queryInput.value = "";
  ask(query);
});

for (const button of document.querySelectorAll(".examples button")) {
  button.addEventListener("click", () => ask(button.textContent));
}

// Link dạng /?q=...&system=graph hỏi ngay khi mở trang (tiện khi trình bày).
loadInfo().then(() => {
  const params = new URLSearchParams(location.search);
  if (params.get("system")) $("system").value = params.get("system");
  if (params.get("q")) ask(params.get("q"));
});
