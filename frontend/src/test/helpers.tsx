import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import { AssistantProvider } from '@/context/AssistantContext'
import { FilterProvider } from '@/context/FilterContext'
import { UiProvider } from '@/context/UiContext'

type Handler = (url: URL, init?: RequestInit) => { status?: number; body: unknown } | Promise<{ status?: number; body: unknown }>

/** A valid /health response. Several components read it, so the mock serves it unless a test supplies its own. */
export const healthyBody = {
  status: 'healthy',
  timestamp_utc: '2026-10-11T12:00:00+00:00',
  version: '0.2.0',
  artifacts_ready: { parquet_data: true, sentiment_models: true, vector_search_faiss: true, nmf_topics: true, distilbert_model: false },
  total_reviews: 100,
  retrieval: { ready: true, indexed_reviews: 100, eligible_reviews: 100, coverage_pct: 100, embedding_model: 'test', built_at: null, limited_by: null },
  assistant: { llm_configured: false, llm_model: 'gemini-test' },
  setup_steps: {} as Record<string, string>,
}

/** Replaces fetch with a function that answers from `handler`. Returns the mock so tests can inspect calls. */
export function mockApi(handler: Handler, options: { serveHealth?: boolean } = {}) {
  const { serveHealth = true } = options
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input))
    if (serveHealth && url.pathname.endsWith('/health')) {
      return new Response(JSON.stringify(healthyBody), { status: 200, headers: { 'Content-Type': 'application/json' } })
    }
    const { status = 200, body } = await handler(url, init)
    return new Response(typeof body === 'string' ? body : JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

export function renderWithProviders(ui: ReactElement, { route = '/' }: { route?: string } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[route]}>
        <FilterProvider>
          <UiProvider>
            <AssistantProvider>{ui}</AssistantProvider>
          </UiProvider>
        </FilterProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}
