"use strict";

const POLL_MS = 1500;
const LABELS = {
  bytedance: "ByteDance (piano, haute résolution)",
  mt3: "MT3 (multi-instrument)",
  basic_pitch: "Basic Pitch (léger, rapide)",
};
const FILE_LABELS = {
  "score.pdf": "PDF",
  "score.musicxml": "MusicXML",
  "score.html": "Aperçu HTML",
  "raw.mid": "MIDI brut",
  "quantized.mid": "MIDI quantifié",
  "cleaned.mid": "MIDI nettoyé",
  "others.mid": "MIDI autres instruments",
  "report.json": "Rapport",
  "job.log": "Journal",
};
const HIDDEN_FILES = new Set(["audio.wav", "raw_notes.json"]);

const $ = (id) => document.getElementById(id);
const form = $("upload-form");
let pollTimer = null;

function showError(element, message) {
  element.textContent = message;
  element.hidden = !message;
}

async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    throw new Error("Serveur injoignable. Vérifiez qu'il est bien lancé.");
  }
  let data = null;
  try {
    data = await response.json();
  } catch (error) {
    /* réponse sans JSON */
  }
  if (!response.ok) {
    throw new Error((data && data.detail) || `Erreur ${response.status}`);
  }
  return data;
}

function engineOption(name, title, detail, available, checked) {
  const label = document.createElement("label");
  label.className = "engine" + (available ? "" : " unavailable");
  const input = document.createElement("input");
  input.type = "radio";
  input.name = "engine";
  input.value = name;
  input.disabled = !available;
  input.checked = checked;
  const text = document.createElement("span");
  const strong = document.createElement("strong");
  strong.textContent = title;
  const badge = document.createElement("span");
  badge.className = "badge " + (available ? "ok" : "ko");
  badge.textContent = available ? "Disponible" : "Indisponible";
  const small = document.createElement("small");
  small.textContent = detail;
  text.append(strong, badge, small);
  label.append(input, text);
  return label;
}

async function loadEngines() {
  const container = $("engines");
  try {
    const data = await api("/api/engines");
    container.textContent = "";
    const available = data.engines.filter((e) => e.available);
    const preferred = available.find((e) => e.name === data.default) || available[0];
    for (const engine of data.engines) {
      container.append(
        engineOption(
          engine.name,
          LABELS[engine.name] || engine.name,
          engine.available ? "" : engine.detail,
          engine.available,
          preferred !== undefined && engine.name === preferred.name
        )
      );
    }
    container.append(
      engineOption(
        "all",
        "Tous les moteurs (comparaison)",
        "Lance les moteurs l'un après l'autre ; les indisponibles sont signalés.",
        available.length > 0,
        false
      )
    );
    $("submit").disabled = available.length === 0;
  } catch (error) {
    container.textContent = "";
    showError($("form-error"), error.message);
  }
}

function setProgress(fraction) {
  const percent = Math.round(fraction * 100);
  $("progress-bar").style.width = percent + "%";
  $("progress").setAttribute("aria-valuenow", String(percent));
}

function summary(report) {
  if (!report) return "";
  const parts = [];
  if (report.tempo) parts.push(`Tempo : ${report.tempo.bpm} à la noire`);
  if (report.key) parts.push(`Tonalité : ${report.key.name}`);
  if (report.time_signature) parts.push(`Métrique : ${report.time_signature}`);
  if (report.measures) parts.push(`${report.measures} mesures`);
  if (report.notes && report.notes.quantized !== undefined) parts.push(`${report.notes.quantized} notes`);
  if (report.total_s) parts.push(`${Math.round(report.total_s)} s de calcul`);
  return parts.join(" · ");
}

async function clientRender(container, musicxmlUrl) {
  // Rendu de secours côté navigateur, seulement si web/vendor/verovio-toolkit-wasm.js est présent.
  if (!window.verovio || !window.verovio.module) return false;
  const xml = await (await fetch(musicxmlUrl)).text();
  await new Promise((resolve) => {
    if (window.verovio.module.calledRun) resolve();
    else window.verovio.module.onRuntimeInitialized = resolve;
  });
  const toolkit = new window.verovio.toolkit();
  toolkit.setOptions({ pageWidth: 2100, pageHeight: 2970, scale: 40, footer: "none" });
  toolkit.loadData(xml);
  const holder = document.createElement("div");
  holder.className = "client-render";
  for (let page = 1; page <= toolkit.getPageCount(); page += 1) {
    const wrapper = document.createElement("div");
    wrapper.innerHTML = toolkit.renderToSVG(page);
    holder.append(wrapper);
  }
  container.append(holder);
  return true;
}

function renderRun(run) {
  const node = $("run-template").content.firstElementChild.cloneNode(true);
  node.querySelector(".run-title").textContent = LABELS[run.engine] || run.engine;
  node.querySelector(".run-summary").textContent = summary(run.report);
  if (run.status === "error") {
    showError(node.querySelector(".run-error"), run.error || "Échec du traitement.");
    return node;
  }
  const warnings = node.querySelector(".run-warnings");
  for (const warning of (run.report && run.report.warnings) || []) {
    const item = document.createElement("li");
    item.textContent = warning;
    warnings.append(item);
  }
  const downloads = node.querySelector(".downloads");
  const order = Object.keys(FILE_LABELS);
  const base = (file) => file.name.split("/").pop();
  const files = run.files
    .filter((file) => !HIDDEN_FILES.has(base(file)))
    .sort((a, b) => order.indexOf(base(a)) - order.indexOf(base(b)));
  for (const file of files) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = file.url + "?download=true";
    link.textContent = FILE_LABELS[base(file)] || base(file);
    item.append(link);
    downloads.append(item);
  }
  const preview = node.querySelector(".preview");
  const html = run.files.find((file) => base(file) === "score.html");
  const musicxml = run.files.find((file) => base(file) === "score.musicxml");
  if (html) {
    const frame = document.createElement("iframe");
    frame.src = html.url;
    frame.title = `Aperçu de la partition (${run.engine})`;
    frame.loading = "lazy";
    preview.append(frame);
  } else if (musicxml) {
    clientRender(preview, musicxml.url).catch(() => {});
  }
  return node;
}

function renderResults(job) {
  const results = $("results");
  results.textContent = "";
  for (const run of job.runs) {
    results.append(renderRun(run));
  }
  results.hidden = false;
}

async function poll(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    setProgress(job.progress);
    $("job-step").textContent =
      job.status === "queued" ? "En attente…" : job.status === "running" ? job.step : "";
    if (job.status === "done" || job.status === "error") {
      setProgress(1);
      $("submit").disabled = false;
      $("job-step").textContent = job.status === "done" ? "Terminé." : "";
      showError($("job-error"), job.status === "error" ? job.error || "Échec du traitement." : "");
      renderResults(job);
      return;
    }
  } catch (error) {
    showError($("job-error"), error.message);
    $("submit").disabled = false;
    return;
  }
  pollTimer = setTimeout(() => poll(jobId), POLL_MS);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError($("form-error"), "");
  showError($("job-error"), "");
  const file = $("file").files[0];
  const engine = form.querySelector('input[name="engine"]:checked');
  if (!file) return showError($("form-error"), "Choisissez un fichier audio.");
  if (!engine) return showError($("form-error"), "Choisissez un moteur.");
  const body = new FormData();
  body.append("file", file);
  body.append("engine", engine.value);
  $("submit").disabled = true;
  clearTimeout(pollTimer);
  try {
    const created = await api("/api/jobs", { method: "POST", body });
    $("job-name").textContent = file.name;
    $("job").hidden = false;
    $("results").hidden = true;
    setProgress(0);
    history.replaceState(null, "", `?job=${created.job_id}`);
    poll(created.job_id);
  } catch (error) {
    showError($("form-error"), error.message);
    $("submit").disabled = false;
  }
});

loadEngines();

// Réouverture d'un job existant : /?job=<identifiant>
const existing = new URLSearchParams(location.search).get("job");
if (existing) {
  api(`/api/jobs/${existing}`)
    .then((job) => {
      $("job-name").textContent = job.filename;
      $("job").hidden = false;
      poll(existing);
    })
    .catch(() => history.replaceState(null, "", location.pathname));
}
