"use strict";

/* Perfect Saturday Planner: front end.
   - Sends the request to /api/plan/stream and draws each agent step as it arrives (NDJSON).
   - Falls back to /api/plan if streaming is unavailable.
   - Builds every node with textContent (no innerHTML), so model or user text can never inject markup. */

(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const INTERESTS = [
    "food", "coffee", "music", "walks", "nature", "art", "history", "books", "markets",
    "shopping", "nightlife", "theatre", "movies", "photography", "wellness", "spirituality", "games", "science",
  ];
  const DIETS = [["none", "No preference"], ["vegetarian", "Vegetarian"], ["vegan", "Vegan"], ["jain", "Jain"]];
  const LIMITS = [["avoid crowded places", "Avoid crowds"], ["no alcohol", "No alcohol"], ["indoor only", "Indoor only"]];
  const TIME_QUICK = ["2 hours", "3 hours", "4 hours", "6 hours"];
  const STARTS = ["9:00 am", "10:00 am", "11:00 am", "12:00 pm", "2:00 pm", "4:00 pm", "5:00 pm", "6:00 pm", "7:00 pm"];
  const FALLBACK_CITIES = ["Bangalore", "Pune", "Gurugram"];
  const MODE_TEXT = { walk: "Walk", "auto/cab": "Auto or cab", "walk/auto": "Walk or auto", stay: "Same place" };
  const CROWD_TEXT = { low: "Quiet", medium: "Moderate crowd", high: "Crowded" };
  const STATE_TEXT = {
    running: "Running", ok: "Done", retry: "Retried after a problem",
    fallback: "Used a fallback", error: "Failed", info: "Note",
  };

  /* ------------------------------------------------------------- helpers */

  /** Tiny DOM builder. Strings become text nodes; attributes are set with setAttribute. */
  function h(tag, attrs, ...kids) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null || v === false) continue;
        if (k === "class") node.className = v;
        else node.setAttribute(k, v === true ? "" : String(v));
      }
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }

  const SVG_NS = "http://www.w3.org/2000/svg";
  const ICONS = {
    ok: ["M3.5 8.5l3 3 6-7"],
    retry: ["M13 8a5 5 0 1 1-1.7-3.8", "M13.2 2.4v3.3H9.9"],
    fallback: ["M3 13V9a3 3 0 0 1 3-3h7", "M10 3l3 3-3 3"],
    error: ["M4.5 4.5l7 7", "M11.5 4.5l-7 7"],
    info: ["M8 7.2v4.6", "M8 4.3v.2"],
    running: ["M8 2.5a5.5 5.5 0 1 0 5.5 5.5"],
  };
  function icon(name) {
    const s = document.createElementNS(SVG_NS, "svg");
    s.setAttribute("viewBox", "0 0 16 16");
    s.setAttribute("aria-hidden", "true");
    s.setAttribute("focusable", "false");
    for (const d of ICONS[name] || ICONS.ok) {
      const p = document.createElementNS(SVG_NS, "path");
      p.setAttribute("d", d);
      p.setAttribute("fill", "none");
      p.setAttribute("stroke", "currentColor");
      p.setAttribute("stroke-width", "2");
      p.setAttribute("stroke-linecap", "round");
      p.setAttribute("stroke-linejoin", "round");
      s.append(p);
    }
    return s;
  }

  const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
  const inr = (n) => "₹" + Math.round(Number(n) || 0).toLocaleString("en-IN");
  function clock(total) {
    const m = ((Math.round(total) % 1440) + 1440) % 1440;
    const hh = Math.floor(m / 60);
    return { t: `${hh % 12 || 12}:${String(m % 60).padStart(2, "0")}`, ap: hh >= 12 ? "pm" : "am" };
  }
  const clock12 = (m) => { const c = clock(m); return `${c.t} ${c.ap}`; };
  function mins(m) {
    m = Math.round(Number(m) || 0);
    if (m < 60) return `${m} min`;
    const hh = Math.floor(m / 60), r = m % 60;
    return r ? `${hh} h ${r} min` : `${hh} h`;
  }
  const fmtMs = (ms) => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`);
  const labelize = (k) => cap(String(k).replace(/_/g, " "));
  const list = (items) => h("ul", null, items.map((t) => h("li", null, t)));

  /* --------------------------------------------------------------- state */

  const form = $("#form");
  const planEl = $("#plan");
  const traceEl = $("#trace");
  const stepsEl = $("#steps");
  const traceFoot = $("#trace-foot");
  const alertEl = $("#alert");
  const formError = $("#form-error");
  const goBtn = $("#go");
  const stopBtn = $("#stop");
  const clearBtn = $("#clear");
  const outputEl = $("#output");
  const announceEl = $("#announce");
  const statusEl = $("#status");

  let controller = null;
  let running = false;
  let cityNames = FALLBACK_CITIES;
  const stepNodes = new Map();

  function announce(text) {
    announceEl.textContent = "";
    window.setTimeout(() => { announceEl.textContent = text; }, 60);
  }

  /* ---------------------------------------------------------------- form */

  function choices(container, type, name, items, checkedValue) {
    for (const [value, label] of items) {
      const input = h("input", { type, name, value });
      if (checkedValue !== undefined && value === checkedValue) input.setAttribute("checked", "");
      container.append(h("label", { class: "chip" }, input, h("span", null, label)));
    }
  }

  function buildForm() {
    choices($("#interests"), "checkbox", "interest", INTERESTS.map((x) => [x, cap(x)]));
    choices($("#diet"), "radio", "diet", DIETS, "none");
    choices($("#limits"), "checkbox", "limit", LIMITS);
    const quick = $("#time-quick");
    for (const t of TIME_QUICK) {
      quick.append(h("button", { type: "button", "data-time": t }, t));
    }
    quick.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-time]");
      if (!b) return;
      $("#time").value = b.getAttribute("data-time");
      $("#time").focus();
    });
    const start = $("#start");
    start.append(h("option", { value: "" }, "Let the agent choose"));
    for (const s of STARTS) start.append(h("option", { value: s }, s));
  }

  function checkedValues(name) {
    return [...form.querySelectorAll(`input[name="${name}"]:checked`)].map((i) => i.value);
  }

  /** Turns the form into the API payload. Returns {payload} or {error}. */
  function collect() {
    const val = (id) => $(id).value.trim();
    const payload = {};
    if (val("#city")) payload.city = val("#city");
    if (val("#budget") !== "") {
      const b = Number(val("#budget"));
      if (!Number.isFinite(b) || b < 0) return { error: "The budget needs to be zero or more." };
      payload.budget = b;
    }
    if (val("#time")) payload.available_time = val("#time");
    if (val("#start")) payload.start_time = val("#start");
    if (val("#mood")) payload.mood = val("#mood");
    const interests = checkedValues("interest");
    if (interests.length) payload.interests = interests;
    const constraints = [];
    const diet = checkedValues("diet")[0];
    if (diet && diet !== "none") constraints.push(diet);
    constraints.push(...checkedValues("limit"));
    for (const part of val("#extra").split(/[,;\n]/)) if (part.trim()) constraints.push(part.trim());
    if (constraints.length) payload.constraints = constraints;
    if (val("#free")) payload.free_text = val("#free");
    if (!Object.keys(payload).length) return { error: "Add a city, or describe your day in your own words." };
    return { payload };
  }

  function showFormError(msg) {
    formError.textContent = msg;
    formError.hidden = false;
  }
  function hideFormError() {
    formError.hidden = true;
    formError.textContent = "";
  }

  function resetForm() {
    form.reset();
    hideFormError();
  }

  /* --------------------------------------------------------------- views */

  function emptyView(title, text) {
    const ghost = h("ol", { class: "ghost", "aria-hidden": "true" });
    for (const food of [false, true, false]) {
      ghost.append(h("li", null,
        h("span", { class: "ghost-time" }),
        h("span", { class: food ? "ghost-ring ghost-ring-food" : "ghost-ring" }),
        h("span", { class: "ghost-text" }, h("i"), h("i"))));
    }
    return h("div", { class: "empty" }, ghost, h("h2", { id: "plan-title", tabindex: "-1" }, title), h("p", null, text));
  }

  function modeView(r) {
    const l = r.llm || {};
    let text;
    if (r.mode === "ai") text = `Planned with an AI model (${l.model || "OpenAI"}) and checked by code.`;
    else if (l.enabled) text = "Basic mode: the AI model could not be used for this plan, so built-in rules did the planning.";
    else text = "Basic mode: no AI key is set on this server, so built-in rules did the planning.";
    return h("div", { class: `mode ${r.mode === "ai" ? "mode-ai" : "mode-basic"}` },
      h("span", { class: "mode-dot", "aria-hidden": "true" }),
      h("div", null, h("p", null, text), (l.notes || []).map((n) => h("p", null, n))));
  }

  function totalsView(r) {
    const t = r.totals;
    if (!t) return null;
    const hasBudget = t.budget != null;
    const pct = hasBudget && t.budget > 0 ? Math.min(100, Math.round((t.total_cost / t.budget) * 100)) : 0;
    let bar = null;
    if (hasBudget && t.budget > 0) {
      const fill = h("span", { class: "budget-fill" });
      fill.style.width = `${pct}%`;
      bar = h("span", { class: "budget-bar", role: "img", "aria-label": `${pct} percent of the budget used` }, fill);
    }
    const cost = h("div", { class: "total" },
      h("span", { class: "num" }, inr(t.total_cost)),
      bar,
      h("span", { class: "cap" }, hasBudget ? `of your ${inr(t.budget)} budget, ${inr(Math.max(0, t.remaining ?? 0))} left` : "estimated total"),
      h("span", { class: "cap" }, `Venues ${inr(t.spend)}, getting around ${inr(t.travel_cost)}`));
    const time = h("div", { class: "total" },
      h("span", { class: "num" }, mins(t.minutes_used)),
      h("span", { class: "cap" }, `out and about, ${clock12(t.start)} to ${clock12(t.end)}`),
      h("span", { class: "cap" }, `${mins(t.minutes_available)} available`));
    const travel = h("div", { class: "total" },
      h("span", { class: "num" }, t.travel_minutes ? mins(t.travel_minutes) : "None"),
      h("span", { class: "cap" }, t.travel_minutes ? "travelling between places" : "no travel needed"),
      h("span", { class: "cap" }, t.travel_cost ? `${inr(t.travel_cost)} on autos and cabs` : "no ride fares counted"));
    return h("div", { class: "totals" }, cost, time, travel);
  }

  function dayBarView(r) {
    const t = r.totals;
    if (!t || !r.stops.length) return null;
    const w0 = t.start;
    const w1 = Math.max(t.start + (t.minutes_available || 0), t.end);
    const span = Math.max(1, w1 - w0);
    const pct = (m) => Math.max(0, Math.min(100, ((m - w0) / span) * 100));
    const track = h("div", { class: "daybar-track" });
    const seg = (cls, a, b) => {
      const el = h("span", { class: `seg ${cls}` });
      el.style.left = `${pct(a)}%`;
      el.style.width = `${Math.max(0.6, pct(b) - pct(a))}%`;
      track.append(el);
    };
    for (const s of r.stops) {
      const trip = s.travel && s.travel.mode !== "start" ? s.travel.minutes : 0;
      const arrive = s.start - (s.wait || 0);
      if (trip > 0) seg("seg-travel", arrive - trip, arrive);
      seg(s.kind === "food" ? "seg-food" : "seg-activity", s.start, s.end);
    }
    const summary = r.stops.map((s) => `${s.name}, ${clock12(s.start)} to ${clock12(s.end)}`).join("; ");
    return h("div", { class: "daybar" },
      h("div", { role: "img", "aria-label": `Your time window from ${clock12(w0)} to ${clock12(w1)}: ${summary}` }, track),
      h("div", { class: "daybar-ends", "aria-hidden": "true" }, h("span", null, clock12(w0)), h("span", null, clock12(w1))),
      h("div", { class: "legend", "aria-hidden": "true" },
        h("span", null, h("i", { class: "swatch swatch-activity" }), "Activity"),
        h("span", null, h("i", { class: "swatch swatch-food" }), "Food"),
        h("span", null, h("i", { class: "swatch swatch-travel" }), "Travel"),
        "Empty space is free time"));
  }

  function crowdView(level) {
    return h("span", { class: "crowd", "data-level": level },
      h("span", { class: "crowd-bars", "aria-hidden": "true" }, h("i"), h("i"), h("i")),
      h("span", null, h("span", { class: "sr-only" }, "Expected crowd: "), CROWD_TEXT[level] || level));
  }

  function stopView(s, i) {
    const food = s.kind === "food";
    const st = clock(s.start);
    const generic = s.source === "generic";
    const where = generic ? null
      : s.category ? `${cap(s.category)}${s.area_name ? ` near ${s.area_name}` : ""}`
        : s.area_name ? `Near ${s.area_name}` : null;
    const price = `${s.cost > 0 ? `About ${inr(s.cost)}` : "Free"}, ${mins(s.duration)} there`;
    const notes = [...(s.warnings || []), generic ? null : s.verify, ...(s.flags || []).map(cap)].filter(Boolean);
    const vegmark = s.diet === "pure_veg"
      ? h("span", { class: "vegmark", role: "img", "aria-label": "Pure vegetarian", title: "Pure vegetarian" })
      : s.diet === "non_veg_focus"
        ? h("span", { class: "vegmark vegmark-non", role: "img", "aria-label": "Serves mostly non-vegetarian food", title: "Serves mostly non-vegetarian food" })
        : null;
    const li = h("li", { class: `stop ${food ? "stop-food" : "stop-activity"}` },
      h("div", { class: "when" },
        h("span", { class: "time" }, st.t, h("span", { class: "ampm" }, st.ap)),
        h("span", { class: "until" }, `until ${clock12(s.end)}`)),
      h("div", { class: "rail", "aria-hidden": "true" }, h("span", { class: "ring" })),
      h("div", { class: "body" },
        h("h3", { class: "stop-name" }, s.name, vegmark),
        where ? h("p", { class: "stop-meta" }, where) : null,
        h("p", { class: "stop-meta" }, price, s.crowd_level && !generic ? crowdView(s.crowd_level) : null),
        s.why ? h("p", { class: "why" }, s.why) : null,
        s.note && !s.why ? h("p", { class: "why" }, s.note) : null,
        notes.length ? list(notes) : null));
    if (notes.length) li.querySelector("ul").className = "stop-notes";
    li.style.setProperty("--i", String(i));
    return li;
  }

  function legView(s, i) {
    const t = s.travel || {};
    const walking = t.mode === "walk" || t.mode === "walk/auto";
    const bits = [`${t.minutes} min`];
    if (t.cost > 0) bits.push(`about ${inr(t.cost)}`);
    if (t.km > 0) bits.push(`${t.km} km`);
    const text = h("div", { class: "leg-text" },
      h("span", null, h("span", { class: "leg-mode" }, MODE_TEXT[t.mode] || cap(t.mode)), `, ${bits.join(", ")}`),
      s.wait > 0 ? h("span", { class: "stop-text-wait" }, `Then about ${s.wait} min of waiting for it to open`) : null);
    const li = h("li", { class: `leg ${walking ? "leg-walk" : "leg-ride"}` }, h("div", { class: "rail", "aria-hidden": "true" }), text);
    li.style.setProperty("--i", String(i));
    return li;
  }

  function routeView(stops) {
    const ol = h("ol", { class: `route${reduceMotion ? "" : " reveal"}`, "aria-label": "Your route, in order" });
    let i = 0;
    stops.forEach((s, n) => {
      const t = s.travel || {};
      if (n > 0 && t.mode !== "stay" && t.mode !== "start") ol.append(legView(s, i++));
      ol.append(stopView(s, i++));
    });
    return ol;
  }

  function block(title, content) {
    return h("section", { class: "block" }, h("h3", { class: "block-title" }, title), content);
  }

  function sourceLine(r) {
    const city = r.city || "this city";
    if (r.data_source === "openstreetmap") return `Venues came from OpenStreetMap contributors, fetched live for ${city}. Hours and prices are estimates, so check them before you go.`;
    if (r.data_source === "generic") return "There was no venue data for this city, so these are general suggestions. Check hours and prices before you go.";
    return `Venue data for ${city} is curated sample data, so check opening hours and prices before you go.`;
  }

  function planView(r) {
    const root = h("article", { class: "plan", "aria-labelledby": "plan-title" });
    root.append(h("h2", { class: "headline", id: "plan-title", tabindex: "-1" }, r.headline || "Your Saturday"));
    if (r.summary && r.status !== "fallback") root.append(h("p", { class: "summary" }, r.summary));
    if (r.status === "fallback") {
      root.append(h("div", { class: "notice" },
        h("p", { class: "notice-title" }, "This is a general outline you can adapt, not a venue-by-venue plan"),
        r.message ? h("p", null, r.message) : null));
    }
    if (r.relaxed && r.relaxed.length) {
      root.append(h("div", { class: "notice" }, h("p", { class: "notice-title" }, "I loosened some of your limits to make this work"), list(r.relaxed)));
    }
    root.append(modeView(r), totalsView(r), dayBarView(r), routeView(r.stops));
    const blocks = h("div", { class: "blocks" });
    if (r.trade_offs && r.trade_offs.length) blocks.append(block("Trade-offs", list(r.trade_offs)));
    if (r.backup) blocks.append(block("If plans change", h("p", null, r.backup)));
    if (r.cautions && r.cautions.length) blocks.append(block("Check before you go", list(r.cautions)));
    if (r.assumptions && r.assumptions.length) blocks.append(block("What I assumed", list(r.assumptions)));
    if (r.questions && r.questions.length) blocks.append(block("To sharpen this plan", list(r.questions)));
    if (blocks.children.length) root.append(blocks);
    root.append(h("p", { class: "source-line" }, sourceLine(r)));
    return root;
  }

  function needsInfoView(r) {
    const root = h("div", { class: "ask-back" });
    root.append(h("h2", { id: "plan-title", tabindex: "-1" }, r.message || "I need one detail before I can plan well."));
    if (r.questions && r.questions.length) root.append(h("ul", { class: "questions" }, r.questions.map((q) => h("li", null, q))));
    const noCity = !(r.preferences && r.preferences.city);
    if (noCity) {
      const chips = h("div", { class: "chips" });
      for (const name of cityNames) {
        chips.append(h("button", { type: "button", "data-city": name }, name));
      }
      chips.addEventListener("click", (e) => {
        const b = e.target.closest("button[data-city]");
        if (!b) return;
        $("#city").value = b.getAttribute("data-city");
        onSubmit();
      });
      root.append(h("div", { class: "pick" }, h("p", null, "Pick a city and I'll plan with what you've entered so far."), chips));
    }
    if (r.assumptions && r.assumptions.length) {
      root.append(h("div", { class: "blocks" }, block("What I understood so far", list(r.assumptions))));
    }
    return root;
  }

  /* --------------------------------------------------------------- trace */

  function detailBody(d) {
    const empty = (v) => v == null || v === "" || (Array.isArray(v) && !v.length) || (typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length);
    if (empty(d)) return null;
    const fmt = (v) => (Array.isArray(v)
      ? v.map(fmt).join(", ")
      : v && typeof v === "object"
        ? Object.entries(v).map(([k, x]) => `${k.replace(/_/g, " ")}: ${fmt(x)}`).join(", ")
        : String(v));
    if (Array.isArray(d)) {
      return h("ul", null, d.filter((x) => !empty(x)).map((x) => (
        x && typeof x === "object"
          ? h("li", { class: x.severity ? `sev-${x.severity}` : null }, x.message || fmt(x))
          : h("li", null, String(x)))));
    }
    if (typeof d === "object") {
      const dl = h("dl");
      for (const [k, v] of Object.entries(d)) {
        if (empty(v)) continue;
        dl.append(h("dt", null, labelize(k)), h("dd", null, fmt(v)));
      }
      return dl.children.length ? dl : null;
    }
    return h("p", null, String(d));
  }

  function fillStep(li, ev) {
    const state = STATE_TEXT[ev.state] ? ev.state : "ok";
    li.className = `step step-${state}`;
    const body = detailBody(ev.detail);
    li.replaceChildren(
      h("span", { class: "glyph" }, icon(state), h("span", { class: "sr-only" }, STATE_TEXT[state])),
      h("div", { class: "step-body" },
        h("div", { class: "step-head" },
          h("span", { class: "tool" }, ev.label || ev.tool),
          state === "running" ? null : h("span", { class: "ms" }, fmtMs(ev.ms || 0))),
        ev.input ? h("p", { class: "step-input" }, `Input: ${ev.input}`) : null,
        ev.summary ? h("p", { class: "step-sum" }, ev.summary) : state === "running" ? h("p", { class: "step-sum" }, "Working") : null,
        body ? h("details", { class: "step-detail" }, h("summary", null, "Details"), body) : null));
  }

  function upsertStep(ev) {
    let li = stepNodes.get(ev.id);
    if (!li) {
      li = h("li", { class: "step" });
      stepNodes.set(ev.id, li);
      stepsEl.append(li);
    }
    fillStep(li, ev);
  }

  /* ----------------------------------------------------------------- run */

  class ApiError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }

  async function httpError(res) {
    let msg = "";
    try { const j = await res.json(); msg = typeof j.error === "string" ? j.error : ""; } catch { /* not JSON */ }
    if (!msg) {
      msg = res.status === 429 ? "You're going a bit fast. Wait a moment and try again."
        : res.status === 503 ? "The planner is busy. Try again in a few seconds."
          : res.status >= 500 ? "Something went wrong on the server. Try again."
            : "The planner did not accept that request.";
    }
    return new ApiError(msg, res.status);
  }

  async function readNdjson(res, onEvent) {
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    const flush = (line) => {
      line = line.trim();
      if (!line) return;
      let ev = null;
      try { ev = JSON.parse(line); } catch { return; }
      onEvent(ev);
    };
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        flush(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
      }
    }
    flush(buf + dec.decode());
  }

  function setBusy(on) {
    running = on;
    planEl.setAttribute("aria-busy", on ? "true" : "false");
    form.setAttribute("aria-busy", on ? "true" : "false");
    goBtn.disabled = on;
    goBtn.textContent = on ? "Planning" : "Plan my Saturday";
    stopBtn.hidden = !on;
  }

  function showAlert(message) {
    alertEl.replaceChildren(h("strong", null, "Planning stopped"), message);
    alertEl.hidden = false;
  }

  function scrollToOutput() {
    const top = outputEl.getBoundingClientRect().top;
    if (top > window.innerHeight * 0.6 || top < 0) {
      outputEl.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    }
  }

  function beginRun() {
    alertEl.hidden = true;
    stepNodes.clear();
    stepsEl.replaceChildren();
    traceFoot.textContent = "";
    traceEl.hidden = false;
    planEl.replaceChildren(emptyView("Planning your Saturday", "The agent's steps appear below as it finishes each one."));
    setBusy(true);
    announce("Planning your Saturday");
    scrollToOutput();
  }

  function showResult(result, trace) {
    for (const ev of trace || []) upsertStep(ev);
    const elapsed = result.elapsed_ms ? `Finished in ${(result.elapsed_ms / 1000).toFixed(1)} s. ` : "";
    const l = result.llm || {};
    traceFoot.textContent = l.calls
      ? `${elapsed}The model made ${l.calls} call${l.calls === 1 ? "" : "s"} and used ${Number(l.tokens || 0).toLocaleString("en-IN")} tokens.`
      : `${elapsed}No model calls were made for this plan.`;
    planEl.replaceChildren(result.status === "needs_info" ? needsInfoView(result) : planView(result));
    setBusy(false);
    const title = $("#plan-title");
    if (title) title.focus({ preventScroll: true });
    announce(result.status === "needs_info" ? (result.message || "I need one more detail.") : `Plan ready: ${result.headline || "your Saturday"}`);
    scrollToOutput();
  }

  function showFailure(message) {
    setBusy(false);
    traceEl.hidden = !stepNodes.size;
    showAlert(message);
    planEl.replaceChildren(emptyView("Your Saturday will appear here", "Change anything in the form and plan again."));
    announce(`Planning stopped. ${message}`);
    alertEl.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest" });
  }

  async function post(url, payload, signal) {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal });
    if (!res.ok) throw await httpError(res);
    return res;
  }

  async function plan(payload) {
    if (controller) controller.abort();
    const mine = new AbortController();
    controller = mine;
    beginRun();
    let finished = false;
    try {
      const res = await post("/api/plan/stream", payload, mine.signal);
      if (res.body && typeof res.body.getReader === "function") {
        await readNdjson(res, (ev) => {
          if (ev.type === "step") upsertStep(ev);
          else if (ev.type === "result") { finished = true; showResult(ev.result, ev.trace); }
          else if (ev.type === "error") { finished = true; showFailure(ev.message || "Something went wrong on the server. Try again."); }
        });
      }
      if (!finished) {
        // The stream closed early (or is unsupported). Ask for the whole plan in one go.
        const res2 = await post("/api/plan", payload, mine.signal);
        const data = await res2.json();
        showResult(data.result, data.trace);
      }
    } catch (err) {
      if (mine.signal.aborted) return;
      showFailure(err instanceof ApiError ? err.message : "The planner could not be reached. Check your connection and try again.");
    } finally {
      if (controller === mine) controller = null;
    }
  }

  function onSubmit() {
    if (running) return;
    const { payload, error } = collect();
    if (error) { showFormError(error); return; }
    hideFormError();
    plan(payload);
  }

  /* ------------------------------------------------------------- startup */

  async function loadStatus() {
    try {
      const res = await fetch("/api/health");
      const j = await res.json();
      statusEl.textContent = j.ai_enabled ? `AI planning is on (${j.model})` : "AI planning is off, so plans use basic mode";
      if (Array.isArray(j.cities) && j.cities.length) cityNames = j.cities.map((c) => c.display || String(c));
    } catch {
      statusEl.textContent = "Planner status unavailable";
    }
  }

  buildForm();
  form.addEventListener("submit", (e) => { e.preventDefault(); onSubmit(); });
  stopBtn.addEventListener("click", () => {
    if (controller) controller.abort();
    controller = null;
    setBusy(false);
    traceEl.hidden = !stepNodes.size;
    planEl.replaceChildren(emptyView("Stopped", "Change anything in the form and plan again."));
    announce("Stopped");
  });
  clearBtn.addEventListener("click", () => { resetForm(); $("#city").focus(); });
  loadStatus();
})();
