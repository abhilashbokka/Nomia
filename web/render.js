// Pure(ish) DOM rendering. Each render* function reads from `state` and rewrites a specific
// region of the page; callbacks for user actions are passed in from app.js so this module stays
// focused on presentation, not behavior.

import { state, filteredItems } from "./state.js";

const ROUTE_BADGE_CLASS = {
  auto: "badge-auto",
  review: "badge-review",
  unsorted: "badge-unsorted",
  other: "badge-other",
};

const ROUTE_LABEL = {
  auto: "Auto",
  review: "Review",
  unsorted: "Unsorted",
  other: "Other",
  left_untouched: "Untouched",
  skip_duplicate: "Duplicate",
  skip_already_organized: "Already organized",
};

let toastTimer = null;

export function showToast(message, kind = "info") {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.hidden = false;
  el.className = "toast" + (kind === "error" ? " is-error" : kind === "success" ? " is-success" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 4000);
}

export function renderSourceFolders(onRemove) {
  const list = document.getElementById("source-folder-list");
  list.innerHTML = "";
  for (const folder of state.config.source_folders) {
    const li = document.createElement("li");
    const span = document.createElement("span");
    span.textContent = folder;
    span.title = folder;
    const btn = document.createElement("button");
    btn.className = "remove-btn";
    btn.textContent = "×";
    btn.setAttribute("aria-label", `Remove ${folder}`);
    btn.onclick = () => onRemove(folder);
    li.append(span, btn);
    list.append(li);
  }
}

export function renderCategories(onChange, onRemove) {
  const list = document.getElementById("category-list");
  list.innerHTML = "";
  for (const cat of state.config.taxonomy) {
    const row = document.createElement("li");
    row.className = "category-row";

    const labelInput = document.createElement("input");
    labelInput.type = "text";
    labelInput.value = cat.label;
    labelInput.title = "Display label";
    labelInput.oninput = () => onChange(cat.key, { label: labelInput.value });

    const folderInput = document.createElement("input");
    folderInput.type = "text";
    folderInput.value = cat.destination_subfolder;
    folderInput.title = "Destination folder";
    folderInput.oninput = () => onChange(cat.key, { destination_subfolder: folderInput.value });

    const removeBtn = document.createElement("button");
    removeBtn.className = "remove-btn";
    removeBtn.textContent = "×";
    removeBtn.onclick = () => onRemove(cat.key);

    row.append(labelInput, folderInput, removeBtn);
    list.append(row);
  }
}

export function renderNamingPresets(onSelectChange) {
  const select = document.getElementById("naming-preset-select");
  select.innerHTML = "";
  for (const preset of state.config.naming_presets) {
    const option = document.createElement("option");
    option.value = preset.key;
    option.textContent = `${preset.label} — ${preset.template}`;
    select.append(option);
  }
  const customOption = document.createElement("option");
  customOption.value = "custom";
  customOption.textContent = "Custom…";
  select.append(customOption);

  select.value = state.config.naming_preset_key;
  select.onchange = () => onSelectChange(select.value);

  const customRow = document.getElementById("custom-template-row");
  const customInput = document.getElementById("custom-template-input");
  customRow.hidden = state.config.naming_preset_key !== "custom";
  customInput.value = state.config.custom_template || "";
}

export function setNamingPreviewText(text) {
  document.getElementById("naming-preview").textContent = text || "(preview unavailable)";
}

export function renderModelSelector(onSelect) {
  const container = document.getElementById("model-selector");
  container.innerHTML = "";
  if (!state.modelsStatus) return;

  for (const model of state.modelsStatus.models) {
    const row = document.createElement("div");
    row.className = "model-option" + (model.is_active ? " is-active" : "") + (!model.pulled ? " is-disabled" : "");

    const label = document.createElement("span");
    label.textContent = model.name + (model.is_active ? " (active)" : "");

    if (model.pulled) {
      const btn = document.createElement("button");
      btn.className = "btn btn-tertiary";
      btn.textContent = model.is_active ? "Active" : "Use";
      btn.disabled = model.is_active;
      btn.onclick = () => onSelect(model.name);
      row.append(label, btn);
    } else {
      const cmd = document.createElement("span");
      cmd.className = "model-pull-cmd";
      cmd.textContent = `ollama pull ${model.name}`;
      cmd.title = "Click to copy";
      cmd.onclick = () => {
        navigator.clipboard?.writeText(`ollama pull ${model.name}`);
        showToast("Copied to clipboard.");
      };
      row.append(label, cmd);
    }
    container.append(row);
  }
}

export function renderToggles() {
  document.getElementById("preserve-source-toggle").checked = !!state.config.preserve_source;
  document.getElementById("keep-dump-toggle").checked = !!state.config.keep_dump_copies;
  document.getElementById("sweep-other-toggle").checked = !!state.config.sweep_other_files;
}

export function renderDestination() {
  document.getElementById("destination-input").value = state.config.destination_root || "";
}

function badgeFor(item) {
  const cls = ROUTE_BADGE_CLASS[item.route] || "badge-neutral";
  const badge = document.createElement("span");
  badge.className = "badge " + cls;
  if (item.confidence !== null && item.confidence !== undefined) {
    badge.textContent = `${ROUTE_LABEL[item.route] || item.route} · ${Math.round(item.confidence * 100)}%`;
  } else {
    badge.textContent = ROUTE_LABEL[item.route] || item.route;
  }
  return badge;
}

export function renderItemList(handlers) {
  const list = document.getElementById("item-list");
  const items = filteredItems();
  list.innerHTML = "";

  if (state.items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = "<p>Add source folders and a destination on the left, then run a dry-run preview to see what Nomia would do.</p>";
    list.append(empty);
    return;
  }

  if (items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = "<p>No items match this filter.</p>";
    list.append(empty);
    return;
  }

  items.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "item-row" + (index === state.selectedIndex ? " is-selected" : "");
    row.dataset.itemId = item.item_id;
    row.onclick = () => handlers.onSelect(index);

    const thumb = document.createElement("img");
    thumb.className = "item-thumb";
    thumb.loading = "lazy";
    thumb.src = item.thumbnail_url;
    thumb.onerror = () => { thumb.style.visibility = "hidden"; };

    const main = document.createElement("div");
    main.className = "item-main";

    const titleRow = document.createElement("div");
    titleRow.className = "item-title-row";

    const isEditing = handlers.editingItemId === item.item_id;
    if (isEditing) {
      const input = document.createElement("input");
      input.className = "item-name-input";
      input.type = "text";
      input.value = item.proposed_name || "";
      input.onclick = (e) => e.stopPropagation();
      input.onkeydown = (e) => {
        if (e.key === "Enter") { handlers.onNameSubmit(item.item_id, input.value); }
        if (e.key === "Escape") { handlers.onNameCancel(); }
      };
      input.onblur = () => handlers.onNameSubmit(item.item_id, input.value);
      titleRow.append(input);
      setTimeout(() => { input.focus(); input.select(); }, 0);
    } else {
      const name = document.createElement("span");
      name.className = "item-name";
      name.textContent = item.proposed_name || item.source_path.split("/").pop();
      titleRow.append(name, badgeFor(item));
    }

    const reason = document.createElement("div");
    reason.className = "item-reason";
    reason.textContent = item.reason || item.error || "";

    const dest = document.createElement("div");
    dest.className = "item-dest";
    dest.textContent = item.proposed_dest_path ? `→ ${item.proposed_dest_path}` : "(left in place)";

    main.append(titleRow, reason, dest);

    const decision = document.createElement("div");
    decision.className = "item-decision";
    decision.textContent = item.user_decision === "confirmed" ? "confirmed" : item.user_decision === "skipped" ? "skipped" : "";

    row.append(thumb, main, decision);
    list.append(row);
  });

  const selectedRow = list.querySelector(".item-row.is-selected");
  selectedRow?.scrollIntoView({ block: "nearest" });
}

export function renderActionBarIdle() {
  document.getElementById("action-bar-status").textContent = state.runId
    ? `Previewing run ${state.runId.slice(0, 8)}… — ${state.items.length} file(s)`
    : "Ready.";
  document.getElementById("action-bar-progress").hidden = true;
  document.getElementById("apply-btn").disabled = state.items.length === 0;
  document.getElementById("scan-btn").disabled = false;
}

export function renderProgress(stage, done, total) {
  const bar = document.getElementById("action-bar-progress");
  bar.hidden = false;
  document.getElementById("scan-btn").disabled = true;
  document.getElementById("apply-btn").disabled = true;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  document.getElementById("progress-fill").style.width = `${pct}%`;
  document.getElementById("progress-label").textContent = `${stage} (${done}/${total})`;
}

export function renderApplySummary(result) {
  const bar = document.getElementById("action-bar-progress");
  bar.hidden = true;
  const status = document.getElementById("action-bar-status");
  const verification = result.verification || {};
  const ok = verification.hash_mismatches ? verification.hash_mismatches.length === 0 : true;
  const parts = [`Applied ${result.applied}`, `${result.failed} failed`, `${result.skipped} skipped`];
  if (!ok) parts.push("⚠ verification issues found");
  status.textContent = parts.join(" · ");
  document.getElementById("scan-btn").disabled = false;
}

export function updateHeaderStatus(text, kind) {
  const el = document.getElementById("header-status");
  el.textContent = text;
  el.className = "header-status" + (kind ? ` is-${kind}` : "");
}
