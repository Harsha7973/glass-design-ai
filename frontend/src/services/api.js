// All backend calls live here. The browser only talks to "/api", which the
// Vite dev server proxies to FastAPI (see vite.config.js). No secrets here.
const API_BASE = '/api'

export const MAX_UPLOAD_MB = 15
export const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

async function parseError(response) {
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') return body.detail
  } catch {
    // fall through to generic message
  }
  return `Request failed (HTTP ${response.status}).`
}

export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`)
    return response.ok
  } catch {
    return false
  }
}

export async function uploadImage(file) {
  const form = new FormData()
  form.append('file', file)
  let response
  try {
    response = await fetch(`${API_BASE}/process`, { method: 'POST', body: form })
  } catch {
    throw new Error('Cannot reach the backend. Is it running on port 8000?')
  }
  if (!response.ok) throw new Error(await parseError(response))
  return response.json()
}

export async function getStatus(jobId) {
  const response = await fetch(`${API_BASE}/status/${encodeURIComponent(jobId)}`)
  if (!response.ok) throw new Error(await parseError(response))
  return response.json()
}
