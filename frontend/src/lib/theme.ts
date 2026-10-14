import type { Sentiment } from '@/api/types'

/** Hex values for places that cannot read CSS variables (Recharts). Keep in sync with src/index.css. */
export const COLORS = {
  positive: '#22c58b',
  neutral: '#f5b82e',
  negative: '#f4507a',
  blue: '#4c7dff',
  teal: '#2dd4bf',
  violet: '#9b6bff',
  pink: '#d46bd6',
  grid: '#1b2550',
  axis: '#8794bd',
  surface: '#0c1331',
} as const

export const TOPIC_COLORS = [COLORS.blue, COLORS.teal, COLORS.neutral, COLORS.pink, COLORS.violet, COLORS.positive] as const

export const SENTIMENT_META: Record<Sentiment, { label: string; color: string; chip: string }> = {
  positive: { label: 'Positive', color: COLORS.positive, chip: 'bg-pos/15 text-pos border-pos/30' },
  neutral: { label: 'Neutral', color: COLORS.neutral, chip: 'bg-neu/15 text-neu border-neu/30' },
  negative: { label: 'Negative', color: COLORS.negative, chip: 'bg-neg/15 text-neg border-neg/30' },
}

export const SENTIMENTS: readonly Sentiment[] = ['positive', 'neutral', 'negative']

export const isSentiment = (v: string): v is Sentiment => v === 'positive' || v === 'neutral' || v === 'negative'
