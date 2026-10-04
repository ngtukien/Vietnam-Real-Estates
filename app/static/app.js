const $ = (id) => document.getElementById(id);
const messages = $("messages");
const form = $("chat-form");
const queryInput = $("query");
const sendButton = $("send");
const FILTERS = ["province_name", "property_type_name", "min_price", "max_price", "min_area", "max_area"];

function addOptions(select, values) {
  for (const value of values) select.add(new Option(value, value));
}

async function loadInfo() {
  try {
    const response = await fetch("/api/info");
    const info = await response.json();
    for (const [key, name] of Object.entries(info.models)) $("model").add(new Option(name, key));
    $("model").value = "rag_kg";
    addOptions($("province_name"), info.provinces);
    addOptions($("property_type_name"), info.property_types);
    const s = info.stats;
    const n = (value) => Number(value).toLocaleString("vi-VN");
    $("stats").textContent = `${n(s.documents)} tin · ${n(s.chunks)} chunk · ${n(s.entities)} thực thể · ${n(s.communities)} cộng đồng`;
  } catch {
    $("stats").textContent = "Không kết nối được server";
  }
}

function readFilters() {
  const filters = {};
  for (const key of FILTERS) {
    const raw = $(key).value.trim();
    if (!raw) continue;
    // Giá nhập theo tỷ, API nhận VND.
    filters[key] = key.endsWith("_price") ? Number(raw) * 1e9 : key.endsWith("_area") ? Number(raw) : raw;
  }
  return filters;
}

function bubble(className, text) {
  const element = document.createElement("div");
  element.className = `bubble ${className}`;
  if (text) element.textContent = text;
  messages.append(element);
  element.scrollIntoView({ behavior: "smooth", block: "end" });
  return element;
}

function renderAnswer(answer) {
  const container = bubble("bot");
  const notes = [];
  if (answer.matched_entities.length) notes.push(`Nhận diện: ${answer.matched_entities.join(", ")}`);
  if (answer.filters_ignored) notes.push("Traditional RAG không áp dụng bộ lọc");
  if (notes.length) {
    const note = document.createElement("p");
    note.className = "note muted";
    note.textContent = notes.join(" · ");
    container.append(note);
  }
  if (!answer.best) {
    container.className = "bubble empty";
    container.textContent = answer.message;
    return;
  }

  const best = answer.best;
  const card = $("listing-card").content.firstElementChild.cloneNode(true);
  const set = (selector, text) => { card.querySelector(selector).textContent = text; };
  set(".type", best.property_type);
  set(".score", best.score);
  set(".title", best.title);
  set(".price", best.price);
  set(".area", best.area);
  const rooms = [best.bedrooms && `${best.bedrooms} PN`, best.bathrooms && `${best.bathrooms} WC`].filter(Boolean);
  if (rooms.length) set(".rooms", rooms.join(" · "));
  else card.querySelector(".rooms-cell").remove();
  set(".address", [best.project, best.address].filter(Boolean).join(" — ") || "Chưa rõ địa chỉ");
  set(".description", best.description);
  if (answer.area) set(".area-note", `Tổng quan khu vực (${answer.area.id}): ${answer.area.summary}`);
  set(".evidence pre", best.evidence);
  set(".meta", `Tin #${best.id}${best.published_at ? " · đăng " + best.published_at : ""} · ${answer.model_name} · ${answer.elapsed_ms} ms`);
  container.append(card);
  container.scrollIntoView({ behavior: "smooth", block: "end" });
}

async function ask(query) {
  $("welcome")?.remove();
  bubble("user", query);
  const typing = bubble("typing", "Đang tìm…");
  sendButton.disabled = true;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, model: $("model").value, filters: readFilters() }),
    });
    const answer = await response.json();
    typing.remove();
    if (answer.error) bubble("error", answer.error);
    else renderAnswer(answer);
  } catch {
    typing.remove();
    bubble("error", "Không kết nối được server. Kiểm tra server.py còn chạy.");
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

loadInfo();
