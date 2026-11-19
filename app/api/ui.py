"""Single-page HTML UI at "/" to try one question and inspect every stage."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HalluGuard</title>
<style>
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6;
--good:#006300;--warn:#8a5a00;--bad:#d03b3b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--good:#0ca30c;--warn:#fab219;--bad:#e66767}}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px} h1{font-size:22px;margin:0} .sub{color:var(--ink2);margin:4px 0 20px}
form{display:grid;gap:12px;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px}
textarea,input,select{width:100%;font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px} label{font-size:13px;color:var(--ink2)}
button{font:inherit;background:var(--accent);color:#fff;border:0;border-radius:6px;padding:10px 16px;cursor:pointer;justify-self:start}
button:disabled{opacity:.6;cursor:wait} .card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:16px}
.badge{display:inline-block;font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;border:1px solid currentColor}
.VERIFIED,.UNCHECKED,.verified_by_evidence{color:var(--good)}
.partially_supported,.consistent_not_evidence_checked,.unchecked,.clarification_requested,.abstained_insufficient_evidence{color:var(--warn)}
.needs_human_review{color:var(--bad)} ul{margin:6px 0 0 18px;padding:0} li{margin:2px 0} .UNCERTAIN{color:var(--warn)} .FAILED,.NEEDS_REVIEW{color:var(--bad)}
.meta{color:var(--muted);font-size:13px} details{border-top:1px solid var(--line);padding:8px 0} summary{cursor:pointer}
pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:var(--bg);border-radius:6px;padding:8px;margin:8px 0 0}
.err{color:var(--bad)}
</style></head><body><main>
<h1>HalluGuard</h1>
<p class="sub">Run one question through a configuration and inspect each stage's output, score, latency and tokens.</p>
<form id="f">
  <div><label for="q">Question</label><textarea id="q" rows="3" required>What happens if you swallow chewing gum?</textarea></div>
  <div class="row">
    <div><label for="m">Model (provider:model)</label><input id="m" placeholder="default from HG_DEFAULT_MODEL"></div>
    <div><label for="c">Configuration</label><select id="c">
      <option>C0</option><option>C1</option><option>C2</option><option>C3</option><option>C4</option><option>C5</option><option selected>C6</option></select></div>
    <div><label for="k">Knowledge-base collection</label><input id="k" value="global"></div>
  </div>
  <button id="b" type="submit">Ask</button>
</form>
<div id="out"></div>
</main>
<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
$("#f").addEventListener("submit", async e => {
  e.preventDefault();
  const b = $("#b"); b.disabled = true; b.textContent = "Running...";
  $("#out").innerHTML = "";
  try {
    const body = {question: $("#q").value, configuration: $("#c").value, collection: $("#k").value || "global"};
    if ($("#m").value.trim()) body.model = $("#m").value.trim();
    const r = await fetch("/ask", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    const conf = d.confidence == null ? "n/a" : d.confidence.toFixed(2);
    const vs = d.verification_status || "";
    const list = (title, items) => items.length ? `<p class="meta"><strong>${title}</strong></p><ul class="meta">${items.join("")}</ul>` : "";
    let html = `<div class="card"><span class="badge ${esc(d.decision)}">${esc(d.decision)}</span>
      ${vs ? `<span class="badge ${esc(vs)}">${esc(vs.replaceAll("_", " "))}</span>` : ""}
      <p>${esc(d.final_answer)}</p>
      <p class="meta">confidence ${conf} &middot; ${(d.latency_ms/1000).toFixed(1)} s &middot; ${d.llm_calls} LLM calls &middot;
      ${d.prompt_tokens + d.completion_tokens} tokens &middot; ${d.n_regenerations} regenerations &middot; run #${d.run_id}
      ${d.review_item_id ? " &middot; review item #" + d.review_item_id : ""}</p>
      <p class="meta">Reasons: ${esc(d.decision_reasons.join("; "))}</p>
      ${d.candidate_answer !== d.final_answer ? `<p class="meta">Withheld candidate answer: ${esc(d.candidate_answer)}</p>` : ""}
      ${list("Detected issues", d.issues.map(i => `<li>[${esc(i.type)}${i.hard ? "" : ", minor"}] ${esc(i.detail)}</li>`))}
      ${list("Remaining uncertainty", d.remaining_uncertainty.map(u => `<li>${esc(u)}</li>`))}
      ${list("Sources", d.sources.map(s => `<li>${esc(s.label)}${s.cited ? " (cited)" : ""}: ${esc(s.title)}
        ${s.source ? "&middot; " + esc(s.source) : ""} &middot; relevance ${Number(s.score).toFixed(2)}
        ${s.last_modified ? "&middot; dated " + esc(String(s.last_modified).slice(0, 10)) : ""}</li>`))}
    </div><div class="card"><strong>Stages</strong>`;
    for (const s of d.stages) {
      html += `<details><summary>${s.order + 1}. ${esc(s.stage)} <span class="meta">${s.latency_ms.toFixed(0)} ms,
        ${s.llm_calls} calls, ${s.prompt_tokens + s.completion_tokens} tokens</span></summary>
        <pre>${esc(JSON.stringify(s.output, null, 2))}</pre></details>`;
    }
    $("#out").innerHTML = html + "</div>";
  } catch (err) {
    $("#out").innerHTML = `<div class="card err">${esc(err.message)}</div>`;
  } finally { b.disabled = false; b.textContent = "Ask"; }
});
</script></body></html>"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """Serve the test page."""
    return HTMLResponse(PAGE)
