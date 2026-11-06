/* PadiScan — logika frontend (vanilla JS). Fokus tunggal: unggah gambar -> klasifikasi -> tampilkan hasil. */
(() => {
  "use strict";

  const API_BASE = window.API_BASE || ""; // kosong = origin yang sama (disajikan FastAPI)
  const MAX_MB = 10;
  const ALLOWED = ["image/jpeg", "image/png", "image/webp"];

  const $ = (id) => document.getElementById(id);
  const fileInput = $("fileInput"), dropzone = $("dropzone");
  const btnScan = $("btnScan");

  let diseases = {};
  let file = null;
  let busy = false;
  let previewUrl = null;
  let current = null; // hasil prediksi terakhir (untuk tab detail)

  /* ---------- helpers ---------- */
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtSize = (b) => (b < 1048576 ? (b / 1024).toFixed(0) + " KB" : (b / 1048576).toFixed(1) + " MB");
  const pct = (v) => (v * 100).toFixed(v >= 0.9995 ? 0 : 1) + "%";
  const SEV = { none: "Tidak ada", medium: "Sedang", high: "Tinggi" };
  const SEV_PIPS = { none: "", medium: "s2", high: "s3" };

  function badgeFor(info, low) {
    if (low) return { cls: "unsure", text: "Tidak yakin" };
    if (info.severity === "none") return { cls: "healthy", text: "Sehat" };
    if (info.severity === "medium") return { cls: "mild", text: "Terdeteksi" };
    return { cls: "", text: "Terdeteksi" };
  }
  const prettyModel = (n) => (n.includes("base") ? "ViT-B/16" : n.includes("small") ? "ViT-S/16" : n);

  /* ---------- status model & daftar penyakit ---------- */
  async function init() {
    try {
      diseases = (await (await fetch(API_BASE + "/diseases")).json()).diseases || {};
    } catch { /* grid kosong bila API mati */ }
    renderDiseaseGrid();
    checkHealth();
  }

  async function checkHealth() {
    const st = $("modelStatus"), tx = $("statusText");
    try {
      const h = await (await fetch(API_BASE + "/health")).json();
      if (h.model_loaded) {
        st.className = "status ok";
        const acc = h.model_info && h.model_info.test_accuracy;
        tx.textContent = "Model siap" + (acc ? ` · akurasi uji ${(acc * 100).toFixed(1)}%` : "");
        if (h.model_info && h.model_info.model_name) $("modelChip").textContent = prettyModel(h.model_info.model_name);
      } else {
        st.className = "status bad";
        tx.textContent = "Model belum dimuat";
      }
    } catch {
      st.className = "status bad";
      tx.textContent = "Server tidak terhubung";
    }
  }

  /* ---------- pemilihan file ---------- */
  function showError(msg) {
    const el = $("errorMsg");
    el.textContent = msg;
    el.hidden = !msg;
  }

  function setFile(f) {
    if (!f) return;
    showError("");
    if (!ALLOWED.includes(f.type)) return showError("Format tidak didukung. Gunakan JPG, PNG, atau WebP.");
    if (f.size > MAX_MB * 1048576) return showError(`Ukuran file melebihi ${MAX_MB} MB.`);

    file = f;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(f);
    $("previewImg").src = previewUrl;
    $("dzEmpty").hidden = true;
    $("dzPreview").hidden = false;
    dropzone.classList.add("has-file");
    $("fileName").textContent = f.name;
    $("fileSize").textContent = fmtSize(f.size);
    $("fileMeta").hidden = false;
    resetResult();
    updateScan();
  }

  function clearFile() {
    file = null;
    fileInput.value = "";
    $("dzEmpty").hidden = false;
    $("dzPreview").hidden = true;
    dropzone.classList.remove("has-file", "scanning");
    $("fileMeta").hidden = true;
    updateScan();
  }

  function updateScan() { btnScan.disabled = busy || !file; }

  dropzone.addEventListener("click", () => { if (!file) fileInput.click(); });
  dropzone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !file) { e.preventDefault(); fileInput.click(); }
  });
  $("btnChange").addEventListener("click", () => { clearFile(); fileInput.click(); });
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

  // drag & drop (seluruh halaman)
  let depth = 0;
  const hasFiles = (e) => e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
  window.addEventListener("dragenter", (e) => { if (hasFiles(e)) { depth++; dropzone.classList.add("drag"); } });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) dropzone.classList.remove("drag"); });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); depth = 0; dropzone.classList.remove("drag");
    setFile(e.dataTransfer.files[0]);
    $("deteksi").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  // tempel gambar dari clipboard
  window.addEventListener("paste", (e) => {
    const item = Array.from(e.clipboardData?.items || []).find((i) => i.type.startsWith("image/"));
    if (item) setFile(item.getAsFile());
  });

  /* ---------- klasifikasi ---------- */
  function setBusy(v) {
    busy = v;
    $("btnSpinner").hidden = !v;
    $("progress").hidden = !v;
    $("btnScanLabel").textContent = v ? "Menganalisis…" : "Scan Sekarang";
    dropzone.classList.toggle("scanning", v);
    updateScan();
  }

  btnScan.addEventListener("click", async () => {
    if (!file || busy) return;
    showError("");
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(API_BASE + "/predict", { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Terjadi kesalahan (${res.status}).`);
      renderResult(data);
    } catch (err) {
      const offline = err instanceof TypeError;
      showError(offline ? "Tidak dapat terhubung ke server. Pastikan backend berjalan." : err.message);
    } finally {
      setBusy(false);
    }
  });

  /* ---------- tampilan hasil ---------- */
  function resetResult() {
    current = null;
    $("resultBody").hidden = true;
    $("resultEmpty").hidden = false;
  }

  function renderResult(r) {
    current = r;
    const info = r.info || {};
    const b = badgeFor(info, r.low_confidence);

    $("resultEmpty").hidden = true;
    $("resultBody").hidden = false;

    const badge = $("resBadge");
    badge.className = "badge " + b.cls;
    badge.textContent = b.text;
    $("resName").textContent = info.name || r.label;
    $("resType").textContent = info.type && info.type !== "-" ? info.type : "";

    const warn = $("warnBox");
    warn.hidden = !r.low_confidence;
    if (r.low_confidence) {
      $("warnText").textContent = `Tingkat keyakinan model rendah (${pct(r.confidence)}). Coba foto ulang dengan pencahayaan baik, fokus pada satu daun, atau pastikan objek adalah daun padi.`;
    }

    // confidence: angka + bar (animasi)
    const bar = $("resConfBar");
    bar.className = r.low_confidence ? "low" : "";
    bar.style.width = "0";
    countUp($("resConf"), r.confidence);
    requestAnimationFrame(() => requestAnimationFrame(() => (bar.style.width = (r.confidence * 100).toFixed(1) + "%")));

    // keparahan
    $("resSevLabel").textContent = SEV[info.severity] || "—";
    $("resPips").className = "pips " + (SEV_PIPS[info.severity] || "");

    // top-3
    $("resTop").innerHTML = r.top.map((t, i) => `
      <div class="bar-row ${i ? "rest" : ""}">
        <span>${esc(t.name)}</span><span class="pct">${pct(t.confidence)}</span>
        <div class="track"><div class="fill" data-w="${(t.confidence * 100).toFixed(1)}"></div></div>
      </div>`).join("");
    requestAnimationFrame(() => requestAnimationFrame(() =>
      document.querySelectorAll("#resTop .fill").forEach((f) => (f.style.width = f.dataset.w + "%"))));

    selectTab("deskripsi");
    if (window.matchMedia("(max-width: 960px)").matches) {
      $("resultBody").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function countUp(el, target) {
    const t0 = performance.now(), dur = 900;
    const tick = (t) => {
      const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = pct(target * e);
      if (k < 1) requestAnimationFrame(tick);
      else el.textContent = pct(target);
    };
    requestAnimationFrame(tick);
  }

  /* ---------- tab detail ---------- */
  const ul = (arr, ordered) => {
    const tag = ordered ? "ol" : "ul";
    return arr && arr.length ? `<${tag}>${arr.map((x) => `<li>${esc(x)}</li>`).join("")}</${tag}>` : "<p class=\"muted\">Informasi belum tersedia.</p>";
  };

  function selectTab(name) {
    const info = (current && current.info) || {};
    document.querySelectorAll(".tabs .tab").forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on);
      if (on) moveIndicator(t);
    });
    const panel = $("tabPanel");
    let html = "";
    if (name === "deskripsi") {
      html = `<p>${esc(info.description || "Informasi belum tersedia.")}</p>` +
        (info.cause && info.cause !== "-" ? `<p class="cause">Penyebab: <b>${esc(info.cause)}</b></p>` : "");
    } else if (name === "gejala") html = ul(info.symptoms, false);
    else html = ul(info.treatment, true);
    panel.style.animation = "none"; void panel.offsetWidth; panel.style.animation = "";
    panel.innerHTML = html;
  }

  function moveIndicator(tab) {
    const ind = $("tabIndicator");
    ind.style.width = tab.offsetWidth + "px";
    ind.style.transform = `translateX(${tab.offsetLeft}px)`;
  }

  document.querySelectorAll(".tabs .tab").forEach((t) => t.addEventListener("click", () => selectTab(t.dataset.tab)));
  window.addEventListener("resize", () => {
    const a = document.querySelector(".tabs .tab.active");
    if (a && !$("resultBody").hidden) moveIndicator(a);
  });

  /* ---------- grid kelas penyakit + dialog ---------- */
  function renderDiseaseGrid() {
    const grid = $("diseaseGrid");
    grid.innerHTML = "";
    Object.entries(diseases).forEach(([key, d], i) => {
      const b = badgeFor(d, false);
      const el = document.createElement("button");
      el.type = "button";
      el.className = "d-card reveal";
      el.id = "card-" + key;
      el.style.transitionDelay = (i % 5) * 60 + "ms";
      el.innerHTML = `<span class="badge ${b.cls === "" ? "" : b.cls}">${esc(d.severity === "none" ? "Sehat" : d.severity === "high" ? "Keparahan tinggi" : "Keparahan sedang")}</span>
        <span class="d-name">${esc(d.name)}</span><span class="d-type">${esc(d.type)}</span>`;
      el.addEventListener("click", () => openDialog(d));
      grid.appendChild(el);
    });
    const io = new IntersectionObserver((entries) => entries.forEach((en) => {
      if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
    }), { threshold: 0.15 });
    grid.querySelectorAll(".reveal").forEach((c) => io.observe(c));
  }

  function openDialog(d) {
    const b = badgeFor(d, false);
    const badge = $("dlgBadge");
    badge.className = "badge " + b.cls;
    badge.textContent = d.severity === "none" ? "Sehat" : "Keparahan " + (SEV[d.severity] || "").toLowerCase();
    $("dlgTitle").textContent = d.name;
    $("dlgType").textContent = d.type && d.type !== "-" ? d.type : "";
    $("dlgBody").innerHTML = `<div class="tab-panel">
      <p>${esc(d.description)}</p>
      ${d.cause && d.cause !== "-" ? `<p class="cause">Penyebab: <b>${esc(d.cause)}</b></p>` : ""}
      <h5>Gejala khas</h5>${ul(d.symptoms, false)}
      <h5>Saran penanganan</h5>${ul(d.treatment, true)}
    </div>`;
    $("diseaseDialog").showModal();
  }
  $("dlgClose").addEventListener("click", () => $("diseaseDialog").close());
  $("diseaseDialog").addEventListener("click", (e) => { if (e.target === $("diseaseDialog")) $("diseaseDialog").close(); });

  init();
})();
