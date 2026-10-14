import { keepPreviousData, useMutation, useQuery } from '@tanstack/react-query'
import { ApiError } from './client'
import {
  getCategories,
  getCategoryMetrics,
  getHealth,
  getModelMetrics,
  getMonitoringStatus,
  getOverview,
  getRatingDistribution,
  getReviews,
  getSentimentTrend,
  getTopics,
  predictSentiment,
} from './endpoints'
import type { PredictRequest, ReviewsParams, ScopeParams, TopicsParams } from './types'

/** Hooks that every screen shares. Each key includes the filters, so changing a filter always refetches. */

const FIVE_MINUTES = 5 * 60 * 1000

export const useHealth = () =>
  useQuery({
    queryKey: ['health'],
    queryFn: async ({ signal }) => {
      // The latency shown in the UI is this round trip as measured by the browser, not a server metric.
      const startedAt = performance.now()
      const health = await getHealth(signal)
      return { health, latencyMs: Math.round(performance.now() - startedAt), checkedAt: Date.now() }
    },
    refetchInterval: 30_000,
    staleTime: 15_000,
  })

export const useCategories = () => useQuery({ queryKey: ['categories'], queryFn: ({ signal }) => getCategories(signal), staleTime: FIVE_MINUTES })

export const useOverview = (scope: ScopeParams) =>
  useQuery({ queryKey: ['overview', scope], queryFn: ({ signal }) => getOverview(scope, signal), staleTime: FIVE_MINUTES, placeholderData: keepPreviousData })

export const useRatingDistribution = (scope: ScopeParams) =>
  useQuery({ queryKey: ['rating-distribution', scope], queryFn: ({ signal }) => getRatingDistribution(scope, signal), staleTime: FIVE_MINUTES, placeholderData: keepPreviousData })

export const useSentimentTrend = (scope: ScopeParams) =>
  useQuery({ queryKey: ['trend', scope], queryFn: ({ signal }) => getSentimentTrend(scope, signal), staleTime: FIVE_MINUTES, placeholderData: keepPreviousData })

export const useTopics = (params: TopicsParams = {}) =>
  useQuery({ queryKey: ['topics', params], queryFn: ({ signal }) => getTopics(params, signal), staleTime: FIVE_MINUTES, placeholderData: keepPreviousData })

export const useReviews = (params: ReviewsParams, enabled = true) =>
  useQuery({
    queryKey: ['reviews', params],
    queryFn: ({ signal }) => getReviews(params, signal),
    enabled,
    staleTime: 60_000,
    // Keep the old page on screen while the next one loads so pagination does not flash.
    placeholderData: keepPreviousData,
  })

export const useModelMetrics = (category?: string | null) =>
  useQuery({ queryKey: ['model-metrics', category ?? null], queryFn: ({ signal }) => getModelMetrics(category, signal), staleTime: FIVE_MINUTES })

export const useCategoryMetrics = () => useQuery({ queryKey: ['category-metrics'], queryFn: ({ signal }) => getCategoryMetrics(signal), staleTime: FIVE_MINUTES })

export const useMonitoring = () =>
  useQuery({
    queryKey: ['monitoring'],
    queryFn: ({ signal }) => getMonitoringStatus(signal),
    staleTime: 60_000,
    refetchInterval: 120_000,
  })

export const usePredict = () => useMutation({ mutationFn: (body: PredictRequest) => predictSentiment(body) })

/** Only a lost connection is worth retrying. A 503 or 422 will not fix itself in a few seconds. */
export const shouldRetry = (failureCount: number, error: unknown) => error instanceof ApiError && error.kind === 'network' && failureCount < 2
