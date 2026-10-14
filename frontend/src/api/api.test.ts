import { afterEach, describe, expect, it, vi } from 'vitest'
import { mockApi } from '@/test/helpers'
import { ApiError, buildUrl, request } from './client'
import { getModelMetrics, getOverview, getRatingDistribution, getReviews, getSentimentTrend, getTopics } from './endpoints'

afterEach(() => vi.unstubAllGlobals())

describe('buildUrl', () => {
  it('resolves the same-origin base URL and skips empty parameters', () => {
    const url = new URL(buildUrl('/reviews', { page: 2, category: null, search: '', sentiment: undefined, min_rating: 1 }))
    expect(url.origin).toBe(window.location.origin)
    expect(url.pathname).toBe('/api/v1/reviews')
    expect(url.searchParams.get('page')).toBe('2')
    expect(url.searchParams.get('min_rating')).toBe('1')
    expect([...url.searchParams.keys()]).toEqual(['page', 'min_rating'])
  })

  it('encodes characters that are special in URLs', () => {
    const url = new URL(buildUrl('/reviews', { search: "50% off & it's great" }))
    expect(url.searchParams.get('search')).toBe("50% off & it's great")
    expect(url.search).not.toContain('& it')
  })
})

describe('request', () => {
  it('returns parsed JSON on success', async () => {
    mockApi(() => ({ body: { status: 'ok', items: [1, 2] } }))
    await expect(request('/anything')).resolves.toEqual({ status: 'ok', items: [1, 2] })
  })

  it('maps 503 to an "unavailable" error with the backend message', async () => {
    mockApi(() => ({ status: 503, body: { detail: 'Topic models not yet trained.' } }))
    const error = await request('/topics').catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ kind: 'unavailable', status: 503, message: 'Topic models not yet trained.' })
  })

  it('joins FastAPI validation messages from a 422', async () => {
    mockApi(() => ({ status: 422, body: { detail: [{ msg: 'String should have at least 2 characters' }, { msg: 'Field required' }] } }))
    const error = await request('/predict', { method: 'POST', body: { text: 'a' } }).catch((e: unknown) => e)
    expect(error).toMatchObject({ kind: 'validation', message: 'String should have at least 2 characters; Field required' })
  })

  it('keeps the backend message when a filter value is rejected', async () => {
    mockApi(() => ({ status: 422, body: { detail: "Unknown category 'zzz'. Valid categories: Automotive, Electronics." } }))
    const error = await request('/reviews', { query: { category: 'zzz' } }).catch((e: unknown) => e)
    expect(error).toMatchObject({ kind: 'validation', message: expect.stringContaining('Valid categories') })
  })

  it('falls back to a generic message when the error body is not JSON', async () => {
    mockApi(() => ({ status: 500, body: 'Internal Server Error' }))
    const error = await request('/overview').catch((e: unknown) => e)
    expect(error).toMatchObject({ kind: 'server', status: 500 })
    expect((error as Error).message).toContain('500')
  })

  it('reports a network failure without inventing data', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const error = await request('/overview').catch((e: unknown) => e)
    expect(error).toMatchObject({ kind: 'network', status: null })
  })

  it('sends JSON bodies with a Content-Type header', async () => {
    const fetchMock = mockApi(() => ({ body: {} }))
    await request('/predict', { method: 'POST', body: { text: 'hello' } })
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit
    expect(init.method).toBe('POST')
    expect(init.body).toBe('{"text":"hello"}')
    expect(init.headers).toMatchObject({ 'Content-Type': 'application/json' })
  })
})

/** Each filter the UI offers must end up as a query parameter the backend applies in SQL. */
describe('endpoint filters reach the request', () => {
  const lastParams = (fetchMock: ReturnType<typeof mockApi>) => Object.fromEntries(new URL(String(fetchMock.mock.calls.at(-1)?.[0])).searchParams)
  const lastPath = (fetchMock: ReturnType<typeof mockApi>) => new URL(String(fetchMock.mock.calls.at(-1)?.[0])).pathname

  it.each([
    ['getOverview', getOverview, '/api/v1/overview'],
    ['getSentimentTrend', getSentimentTrend, '/api/v1/sentiment/trend'],
    ['getRatingDistribution', getRatingDistribution, '/api/v1/ratings/distribution'],
  ])('%s sends category and dates', async (_name, fn, path) => {
    const fetchMock = mockApi(() => ({ body: {} }))
    await fn({ category: 'Electronics', date_from: '2022-01-01', date_to: '2022-12-31' })
    expect(lastPath(fetchMock)).toBe(path)
    expect(lastParams(fetchMock)).toEqual({ category: 'Electronics', date_from: '2022-01-01', date_to: '2022-12-31' })
  })

  it('omits filters that are not set', async () => {
    const fetchMock = mockApi(() => ({ body: {} }))
    await getOverview({ category: null, date_from: null })
    expect(lastParams(fetchMock)).toEqual({})
  })

  it('getReviews sends every review filter, sorting and pagination', async () => {
    const fetchMock = mockApi(() => ({ body: {} }))
    await getReviews({
      category: 'Pet_Supplies',
      date_from: '2021-05-01',
      date_to: '2022-05-01',
      sentiment: 'negative',
      min_rating: 1,
      max_rating: 2,
      search: 'refund',
      verified_only: true,
      topic_id: 3,
      sort: 'rating',
      order: 'asc',
      page: 4,
      limit: 50,
    })
    expect(lastParams(fetchMock)).toEqual({
      category: 'Pet_Supplies',
      date_from: '2021-05-01',
      date_to: '2022-05-01',
      sentiment: 'negative',
      min_rating: '1',
      max_rating: '2',
      search: 'refund',
      verified_only: 'true',
      topic_id: '3',
      sort: 'rating',
      order: 'asc',
      page: '4',
      limit: '50',
    })
  })

  it('getTopics sends the sentiment filter used for complaint topics', async () => {
    const fetchMock = mockApi(() => ({ body: {} }))
    await getTopics({ category: 'Automotive', sentiment: 'negative', date_from: '2020-01-01' })
    expect(lastPath(fetchMock)).toBe('/api/v1/topics')
    expect(lastParams(fetchMock)).toEqual({ category: 'Automotive', sentiment: 'negative', date_from: '2020-01-01' })
  })

  it('getModelMetrics sends the category for per-category results', async () => {
    const fetchMock = mockApi(() => ({ body: {} }))
    await getModelMetrics('Electronics')
    expect(lastParams(fetchMock)).toEqual({ category: 'Electronics' })
    await getModelMetrics()
    expect(lastParams(fetchMock)).toEqual({})
  })
})
