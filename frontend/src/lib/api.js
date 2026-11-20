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
  } else if (import.meta.env.VITE_API_URL) {
    _baseUrl = import.meta.env.VITE_API_URL;
  } else if (window.location.port === '5173') {
    _baseUrl = 'http://127.0.0.1:8080';
  } else {
    _baseUrl = '';
  }

  return _baseUrl;
}

// The server's errors carry a JSON `detail`; anything else is reported as status + body.
async function fail(response) {
  const text = await response.text().catch(() => response.statusText);
  let detail;
  try { detail = JSON.parse(text).detail; } catch { /* not JSON */ }
  const error = new Error(typeof detail === 'string' ? detail : `${response.status}: ${text}`);
  error.status = response.status;
  throw error;
}

export async function apiPost(path, body) {
  const base = await getBaseUrl();
  const response = await fetch(`${base}${path}`, { method: 'POST', body });
  if (!response.ok) await fail(response);
  return response;
}

export async function apiGet(path) {
  const base = await getBaseUrl();
  const response = await fetch(`${base}${path}`);
  if (!response.ok) await fail(response);
  return response.json();
}

export async function apiDelete(path) {
  const base = await getBaseUrl();
  const response = await fetch(`${base}${path}`, { method: 'DELETE' });
  if (!response.ok) await fail(response);
  return response;
}

// Start a route's work as a server-side job (`?job=1`); resolves to the job's id.
export async function apiJob(path, body) {
  const response = await apiPost(`${path}?job=1`, body);
  return (await response.json()).job_id;
}

// Waits for a job, then resolves to the response its route would have given.
// Rejects with status 409 if it was cancelled, 404 if the server no longer knows it.
export async function apiResult(jobId) {
  const base = await getBaseUrl();
  const response = await fetch(`${base}/jobs/${jobId}/result`);
  if (!response.ok) await fail(response);
  return response;
}

export async function apiEventSource(path) {
  const base = await getBaseUrl();
  return new EventSource(`${base}${path}`);
}
