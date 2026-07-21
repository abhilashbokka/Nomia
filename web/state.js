// In-memory application state. Server-persisted config is the source of truth for settings;
// this just holds the working copy being edited plus the current run's preview items.

export const state = {
  config: null,
  modelsStatus: null,
  runId: null,
  items: [],
  filter: "all",
  search: "",
  selectedIndex: 0,
  applyResult: null,
  lastAppliedRunId: null,
};

export function filteredItems() {
  const query = state.search.trim().toLowerCase();
  return state.items.filter((item) => {
    if (state.filter !== "all" && item.route !== state.filter) return false;
    if (query) {
      const haystack = `${item.proposed_name || ""} ${item.source_path || ""}`.toLowerCase();
      if (!haystack.includes(query)) return false;
    }
    return true;
  });
}

export function findItem(itemId) {
  return state.items.find((item) => item.item_id === itemId);
}
