/** Thin fetch wrapper. Every backend call goes through `request` so errors look the same everywhere. */

/**
 * The API lives at /api/v1 on the same origin as the page. In production FastAPI serves both the React build and
 * the API, and in development the Vite dev server proxies /api to the backend (see vite.config.ts), so no
 * configuration is needed. Set VITE_API_BASE_URL only to point the app at an API on another host.
 */
const DEFAULT_API_BASE_URL = '/api/v1'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL?.trim() || DEFAULT_API_BASE_URL).replace(/\/+$/, '')

export type ApiErrorKind =
  | 'network' // server not reachable / CORS / offline
  | 'unavailable' // 503: the backend is up but an artifact (dataset, model, index) is missing
  | 'validation' // 422: the request did not match the schema
  | 'not_found'
  | 'server' // other 5xx
  | 'client' // other 4xx

export class ApiError extends Error {
  readonly status: number | null
  readonly kind: ApiErrorKind

  constructor(message: string, status: number | null, kind: ApiErrorKind) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.kind = kind
  }
}

type QueryValue = string | number | boolean | null | undefined
export type QueryParams = Record<string, QueryValue>

interface RequestOptions {
  method?: 'GET' | 'POST'
  query?: QueryParams
  body?: unknown
  signal?: AbortSignal
}

export function buildUrl(path: string, query?: QueryParams): string {
  // The second argument lets a relative base such as "/api/v1" resolve against the page's own origin.
  const url = new URL(`${API_BASE_URL}${path}`, window.location.origin)
  for (const [key, value] of Object.entries(query ?? {})) {
    // Empty strings are treated as "no filter" so callers can pass raw input values.
    if (value === undefined || value === null || value === '') continue
    url.searchParams.set(key, String(value))
  }
  return url.toString()
}

/** FastAPI sends `detail` as a string (HTTPException) or a list of validation problems (422). */
function extractDetail(payload: unknown): string | null {
  if (!payload || typeof payload !== 'object' || !('detail' in payload)) return null
  const { detail } = payload as { detail: unknown }
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((item) => (item && typeof item === 'object' && 'msg' in item ? String((item as { msg: unknown }).msg) : null))
      .filter(Boolean)
      .join('; ')
  }
  return null
}

function kindForStatus(status: number): ApiErrorKind {
  if (status === 503) return 'unavailable'
  if (status === 422) return 'validation'
  if (status === 404) return 'not_found'
  if (status >= 500) return 'server'
  return 'client'
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', query, body, signal } = options
  let response: Response
  try {
    response = await fetch(buildUrl(path, query), {
      method,
      signal,
      headers: body === undefined ? { Accept: 'application/json' } : { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw new ApiError(`Could not reach the ReviewPulse API at ${new URL(API_BASE_URL, window.location.origin).href}. Check that the backend is running.`, null, 'network')
  }

  if (!response.ok) {
    let detail: string | null = null
    try {
      detail = extractDetail(await response.json())
    } catch {
      // Body was not JSON; fall through to the generic message.
    }
    throw new ApiError(detail ?? `The API returned ${response.status} ${response.statusText}.`.trim(), response.status, kindForStatus(response.status))
  }

  try {
    return (await response.json()) as T
  } catch {
    throw new ApiError('The API sent a response that could not be read.', response.status, 'server')
  }
}
