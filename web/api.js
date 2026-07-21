// Thin fetch wrappers over the FastAPI backend. No business logic here - the server is always
// the single source of truth (see CLAUDE.md); this module just shapes HTTP calls.

const BASE = "/api";

async function request(method, path, body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const resp = await fetch(BASE + path, options);
  if (!resp.ok) {
    let detail = "";
    try {
      const data = await resp.json();
      detail = data.detail || JSON.stringify(data);
    } catch {
      detail = await resp.text().catch(() => "");
    }
    throw new Error(detail || `${method} ${path} failed (${resp.status})`);
  }
  const contentType = resp.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return resp.json();
  }
  return resp;
}

export const api = {
  health: () => request("GET", "/health"),
  modelsStatus: () => request("GET", "/models/status"),
  getConfig: () => request("GET", "/config"),
  putConfig: (cfg) => request("PUT", "/config", cfg),
  validatePath: (path) => request("POST", "/validate-path", { path }),
  namingPreview: (template) => request("POST", "/naming/preview", { template }),

  startScan: (body) => request("POST", "/scan", body || {}),
  scanStatus: (runId) => request("GET", `/scan/${runId}/status`),
  getPreview: (runId) => request("GET", `/preview/${runId}`),
  patchItem: (runId, itemId, body) => request("PATCH", `/preview/${runId}/items/${itemId}`, body),
  patchItemsBulk: (runId, body) => request("PATCH", `/preview/${runId}/items:bulk`, body),

  startApply: (runId) => request("POST", `/apply/${runId}`, { confirm: true }),
  applyStatus: (runId) => request("GET", `/apply/${runId}/status`),
  undo: (runId) => request("POST", `/undo/${runId}`),
  lastApplied: () => request("GET", "/runs/last-applied"),

  reportUrl: (runId) => `${BASE}/report/${runId}`,
  thumbnailUrl: (itemId) => `${BASE}/thumbnail/${itemId}`,
};
