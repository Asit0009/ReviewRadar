import { request } from './client'
import type {
  AssistantChatRequest,
  AssistantChatResponse,
  CategoriesResponse,
  CategoryMetricsTable,
  HealthResponse,
  ModelMetricsResponse,
  MonitoringResponse,
  OverviewResponse,
  PredictRequest,
  PredictResponse,
  RatingDistributionResponse,
  ReviewsParams,
  ReviewsResponse,
  ScopeParams,
  TopicsParams,
  TopicsResponse,
  TrendResponse,
} from './types'

/** One function per backend endpoint (see GET /openapi.json). Nothing here reshapes the data. */

export const getHealth = (signal?: AbortSignal) => request<HealthResponse>('/health', { signal })

export const getCategories = (signal?: AbortSignal) => request<CategoriesResponse>('/categories', { signal })

export const getOverview = (scope: ScopeParams = {}, signal?: AbortSignal) => request<OverviewResponse>('/overview', { query: { ...scope }, signal })

export const getSentimentTrend = (scope: ScopeParams = {}, signal?: AbortSignal) => request<TrendResponse>('/sentiment/trend', { query: { ...scope }, signal })

export const getRatingDistribution = (scope: ScopeParams = {}, signal?: AbortSignal) =>
  request<RatingDistributionResponse>('/ratings/distribution', { query: { ...scope }, signal })

export const getTopics = (params: TopicsParams = {}, signal?: AbortSignal) => request<TopicsResponse>('/topics', { query: { ...params }, signal })

export const getReviews = (params: ReviewsParams = {}, signal?: AbortSignal) => request<ReviewsResponse>('/reviews', { query: { ...params }, signal })

export const predictSentiment = (body: PredictRequest, signal?: AbortSignal) =>
  request<PredictResponse>('/predict', { method: 'POST', body, signal })

export const getModelMetrics = (category?: string | null, signal?: AbortSignal) =>
  request<ModelMetricsResponse>('/models/metrics', { query: { category }, signal })

export const getCategoryMetrics = (signal?: AbortSignal) => request<CategoryMetricsTable>('/models/metrics/by-category', { signal })

export const askAssistant = (body: AssistantChatRequest, signal?: AbortSignal) =>
  request<AssistantChatResponse>('/assistant/chat', { method: 'POST', body, signal })

export const getMonitoringStatus = (signal?: AbortSignal) => request<MonitoringResponse>('/monitoring/status', { signal })
