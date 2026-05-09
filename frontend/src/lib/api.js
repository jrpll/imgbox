// Centralised API wrapper.
// In a Tauri context, reads the server config from the Tauri store.
// Falls back to VITE_API_URL or localhost for plain-browser / dev mode.

let _baseUrl = null;

export function resetBaseUrl() {
  _baseUrl = null;
}

async function getBaseUrl() {
  if (_baseUrl) return _baseUrl;

  if (window.__TAURI__) {
    const { invoke } = await import('@tauri-apps/api/core');
    const config = await invoke('get_server_config');
    _baseUrl = config.mode === 'remote' ? config.remote_url : 'http://127.0.0.1:8080';
  } else {
    _baseUrl = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8080';
  }

  return _baseUrl;
}

export async function apiPost(path, body) {
  const base = await getBaseUrl();
  const response = await fetch(`${base}${path}`, { method: 'POST', body });
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new Error(`${response.status}: ${text}`);
  }
  return response;
}

export async function apiGet(path) {
  const base = await getBaseUrl();
  const response = await fetch(`${base}${path}`);
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new Error(`${response.status}: ${text}`);
  }
  return response.json();
}
