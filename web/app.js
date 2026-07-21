import { api } from "./api.js";
import { state, filteredItems, findItem } from "./state.js";
import { setupKeyboardHandler } from "./keyboard.js";
import {
  showToast,
  renderSourceFolders,
  renderCategories,
  renderNamingPresets,
  setNamingPreviewText,
  renderModelSelector,
  renderToggles,
  renderDestination,
  renderItemList,
  renderActionBarIdle,
  renderProgress,
  renderApplySummary,
  updateHeaderStatus,
} from "./render.js";

let editingItemId = null;
let namingPreviewTimer = null;

function refreshItemList() {
  renderItemList({
    onSelect: (index) => { state.selectedIndex = index; editingItemId = null; refreshItemList(); },
    editingItemId,
    onNameSubmit: async (itemId, newName) => {
      editingItemId = null;
      const item = findItem(itemId);
      const previous = item.proposed_name;
      item.proposed_name = newName || previous;
      refreshItemList();
      try {
        await api.patchItem(state.runId, itemId, { name_override: newName || null });
      } catch (err) {
        item.proposed_name = previous;
        refreshItemList();
        showToast(`Could not rename: ${err.message}`, "error");
      }
    },
    onNameCancel: () => { editingItemId = null; refreshItemList(); },
  });
}

async function setItemDecision(item, decision) {
  const previous = item.user_decision;
  item.user_decision = decision;
  refreshItemList();
  try {
    await api.patchItem(state.runId, item.item_id, { user_decision: decision });
  } catch (err) {
    item.user_decision = previous;
    refreshItemList();
    showToast(`Could not update: ${err.message}`, "error");
  }
}

function renderLeftPanelAll() {
  renderSourceFolders(removeSourceFolder);
  renderCategories(onCategoryChange, onCategoryRemove);
  renderNamingPresets(async (key) => {
    state.config.naming_preset_key = key;
    document.getElementById("custom-template-row").hidden = key !== "custom";
    await refreshNamingPreview();
  });
  renderModelSelector(onModelSelect);
  renderToggles();
  renderDestination();
}

async function refreshNamingPreview() {
  const template = state.config.naming_preset_key === "custom"
    ? (document.getElementById("custom-template-input").value || "")
    : (state.config.naming_presets.find((p) => p.key === state.config.naming_preset_key)?.template || "");
  try {
    const result = await api.namingPreview(template);
    setNamingPreviewText(result.example_filename);
  } catch {
    setNamingPreviewText(null);
  }
}

function wireLeftPanel() {
  document.getElementById("add-source-btn").onclick = async () => {
    const input = document.getElementById("source-folder-input");
    const path = input.value.trim();
    if (!path) return;
    const check = await api.validatePath(path);
    if (!check.is_dir) {
      showToast(`'${path}' is not a folder that exists.`, "error");
      return;
    }
    if (!state.config.source_folders.includes(path)) {
      state.config.source_folders.push(path);
    }
    input.value = "";
    renderSourceFolders(removeSourceFolder);
  };

  document.getElementById("destination-input").addEventListener("change", (e) => {
    state.config.destination_root = e.target.value.trim() || null;
  });

  document.getElementById("add-category-btn").onclick = () => {
    const key = `category_${Date.now()}`;
    state.config.taxonomy.push({ key, label: "New category", destination_subfolder: key });
    renderCategories(onCategoryChange, onCategoryRemove);
  };

  document.getElementById("naming-preset-select").onchange = async (e) => {
    state.config.naming_preset_key = e.target.value;
    document.getElementById("custom-template-row").hidden = e.target.value !== "custom";
    await refreshNamingPreview();
  };

  document.getElementById("custom-template-input").addEventListener("input", (e) => {
    state.config.custom_template = e.target.value;
    clearTimeout(namingPreviewTimer);
    namingPreviewTimer = setTimeout(refreshNamingPreview, 150);
  });

  document.getElementById("preserve-source-toggle").onchange = (e) => { state.config.preserve_source = e.target.checked; };
  document.getElementById("keep-dump-toggle").onchange = (e) => { state.config.keep_dump_copies = e.target.checked; };
  document.getElementById("sweep-other-toggle").onchange = (e) => { state.config.sweep_other_files = e.target.checked; };

  document.getElementById("save-config-btn").onclick = async () => {
    try {
      const saved = await api.putConfig(state.config);
      state.config = saved;
      renderLeftPanelAll();
      document.getElementById("save-config-hint").textContent = "Saved.";
      setTimeout(() => { document.getElementById("save-config-hint").textContent = ""; }, 2000);
    } catch (err) {
      showToast(`Could not save settings: ${err.message}`, "error");
    }
  };
}

function removeSourceFolder(folder) {
  state.config.source_folders = state.config.source_folders.filter((f) => f !== folder);
  renderSourceFolders(removeSourceFolder);
}

function onCategoryChange(key, patch) {
  const cat = state.config.taxonomy.find((c) => c.key === key);
  if (cat) Object.assign(cat, patch);
}

function onCategoryRemove(key) {
  state.config.taxonomy = state.config.taxonomy.filter((c) => c.key !== key);
  renderCategories(onCategoryChange, onCategoryRemove);
}

async function onModelSelect(modelName) {
  state.config.model.active_model = modelName;
  renderModelSelector(onModelSelect);
}

async function runScan() {
  try {
    await api.putConfig(state.config);
  } catch (err) {
    showToast(`Could not save settings before scanning: ${err.message}`, "error");
    return;
  }

  let started;
  try {
    started = await api.startScan({});
  } catch (err) {
    showToast(err.message, "error");
    return;
  }

  state.runId = started.run_id;
  document.getElementById("report-link").hidden = true;
  await pollScanWithProgress();
}

async function pollScanWithProgress() {
  for (;;) {
    let status;
    try {
      status = await api.scanStatus(state.runId);
    } catch (err) {
      showToast(err.message, "error");
      renderActionBarIdle();
      return;
    }
    if (status.status === "running") {
      renderProgress(status.stage || "scanning", status.done, status.total);
      await new Promise((r) => setTimeout(r, 200));
      continue;
    }
    if (status.status === "error") {
      showToast(status.error || "Scan failed", "error");
      renderActionBarIdle();
      return;
    }
    break;
  }

  const preview = await api.getPreview(state.runId);
  state.items = preview.items;
  state.selectedIndex = 0;
  editingItemId = null;
  refreshItemList();
  renderActionBarIdle();
}

async function runApply() {
  if (!state.runId || state.items.length === 0) return;
  const confirmed = window.confirm("Apply this plan? Files will be moved/copied according to the preview above.");
  if (!confirmed) return;

  try {
    await api.startApply(state.runId);
  } catch (err) {
    showToast(err.message, "error");
    return;
  }

  for (;;) {
    let status;
    try {
      status = await api.applyStatus(state.runId);
    } catch (err) {
      showToast(err.message, "error");
      return;
    }
    if (status.status === "running") {
      renderProgress(status.stage || "applying", status.done, status.total);
      await new Promise((r) => setTimeout(r, 200));
      continue;
    }
    if (status.status === "error") {
      showToast(status.error || "Apply failed", "error");
      renderActionBarIdle();
      return;
    }
    renderApplySummary(status.result);
    const reportLink = document.getElementById("report-link");
    reportLink.href = api.reportUrl(state.runId);
    reportLink.hidden = false;
    document.getElementById("undo-last-btn").hidden = false;
    document.getElementById("undo-last-btn").dataset.runId = state.runId;
    state.lastAppliedRunId = state.runId;
    showToast("Applied successfully.", "success");
    break;
  }
}

async function runUndo(runId) {
  if (!runId) return;
  const confirmed = window.confirm("Undo this run? Files will be moved back and organized copies removed.");
  if (!confirmed) return;
  try {
    const result = await api.undo(runId);
    showToast(`Undone ${result.undone} file(s)${result.skipped ? `, ${result.skipped} skipped` : ""}.`, "success");
    document.getElementById("undo-last-btn").hidden = true;
  } catch (err) {
    showToast(err.message, "error");
  }
}

function wireActionBar() {
  document.getElementById("scan-btn").onclick = runScan;
  document.getElementById("apply-btn").onclick = runApply;
  document.getElementById("undo-last-btn").onclick = (e) => runUndo(e.target.dataset.runId);
}

function wireFilters() {
  document.getElementById("filter-chips").addEventListener("click", (e) => {
    const btn = e.target.closest(".chip");
    if (!btn) return;
    state.filter = btn.dataset.filter;
    document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("is-active", c === btn));
    state.selectedIndex = 0;
    refreshItemList();
  });
  document.getElementById("search-input").addEventListener("input", (e) => {
    state.search = e.target.value;
    state.selectedIndex = 0;
    refreshItemList();
  });
}

function wireKeyboard() {
  setupKeyboardHandler({
    onNavigate: (delta) => {
      const items = filteredItems();
      if (items.length === 0) return;
      state.selectedIndex = Math.max(0, Math.min(items.length - 1, state.selectedIndex + delta));
      editingItemId = null;
      refreshItemList();
    },
    onConfirm: () => {
      const items = filteredItems();
      const item = items[state.selectedIndex];
      if (item) setItemDecision(item, "confirmed");
    },
    onSkip: () => {
      const items = filteredItems();
      const item = items[state.selectedIndex];
      if (item) setItemDecision(item, "skipped");
    },
    onEdit: () => {
      const items = filteredItems();
      const item = items[state.selectedIndex];
      if (item) { editingItemId = item.item_id; refreshItemList(); }
    },
  });
}

async function init() {
  try {
    const [cfg, modelsStatus, health, lastApplied] = await Promise.all([
      api.getConfig(),
      api.modelsStatus(),
      api.health(),
      api.lastApplied(),
    ]);
    state.config = cfg;
    state.modelsStatus = modelsStatus;
    state.lastAppliedRunId = lastApplied.run_id;

    updateHeaderStatus(
      health.ollama_reachable ? "Ollama connected" : "Ollama unreachable",
      health.ollama_reachable ? null : "error",
    );

    if (state.lastAppliedRunId) {
      document.getElementById("undo-last-btn").hidden = false;
      document.getElementById("undo-last-btn").dataset.runId = state.lastAppliedRunId;
    }

    renderLeftPanelAll();
    await refreshNamingPreview();
    wireLeftPanel();
    wireActionBar();
    wireFilters();
    wireKeyboard();
    refreshItemList();
    renderActionBarIdle();
  } catch (err) {
    showToast(`Could not load Nomia: ${err.message}`, "error");
  }
}

init();
