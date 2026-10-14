import { describe, expect, it } from 'vitest'
import type { ModelMetricsPayload, TopicItem } from '@/api/types'
import { buildCloudWords, topicName } from '@/components/charts/TopicViews'
import { categoryLabel, fmtDate, fmtMonthShort, fmtRatio } from './format'
import { bestByMacroF1, buildModelRows } from './models'
import { cleanReviewText, splitByTerm, truncate } from './text'

describe('cleanReviewText', () => {
  it('turns <br /> into spaces and drops other tags', () => {
    expect(cleanReviewText('Great<br /><br />sound <b>quality</b>')).toBe('Great sound quality')
  })
  it('handles empty input', () => {
    expect(cleanReviewText(null)).toBe('')
    expect(cleanReviewText(undefined)).toBe('')
  })
  it('keeps text that merely looks like markup safe as plain text', () => {
    expect(cleanReviewText('5 < 6 and 7 > 3')).toBe('5 < 6 and 7 > 3')
  })
})

describe('splitByTerm', () => {
  it('splits around case-insensitive matches', () => {
    expect(splitByTerm('Battery died. battery bad', 'battery')).toEqual([
      { text: 'Battery', match: true },
      { text: ' died. ', match: false },
      { text: 'battery', match: true },
      { text: ' bad', match: false },
    ])
  })
  it('treats regex characters in the search term literally', () => {
    expect(splitByTerm('cost (usd) $5', '(usd)')).toEqual([
      { text: 'cost ', match: false },
      { text: '(usd)', match: true },
      { text: ' $5', match: false },
    ])
  })
  it('returns the text untouched without a term', () => {
    expect(splitByTerm('hello', '  ')).toEqual([{ text: 'hello', match: false }])
  })
})

describe('formatting', () => {
  it('formats dates without shifting the day', () => {
    expect(fmtDate('2023-03-01')).toBe('Mar 1, 2023')
  })
  it('formats chart months', () => {
    expect(fmtMonthShort('2023-03')).toBe("Mar '23")
  })
  it('formats category names and ratios', () => {
    expect(categoryLabel('Arts_Crafts_and_Sewing')).toBe('Arts Crafts and Sewing')
    expect(fmtRatio(0.8396)).toBe('84.0%')
  })
  it('truncates long text', () => {
    expect(truncate('abcdefghij', 5)).toBe('abcde…')
    expect(truncate('abc', 5)).toBe('abc')
  })
})

const payload: ModelMetricsPayload = {
  models: {
    logistic_regression: {
      test: {
        accuracy: 0.84,
        macro_f1: 0.62,
        weighted_f1: 0.85,
        negative_precision: 0.5,
        neutral_precision: 0.3,
        positive_precision: 0.7,
        negative_recall: 0.6,
        neutral_recall: 0.3,
        positive_recall: 0.9,
        confusion_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      },
      validation: { accuracy: 0.85, macro_f1: 0.63 },
    },
    linear_svc: { test: { accuracy: 0.87, macro_f1: 0.6 } },
    distilbert: {
      training: { train_samples: 6000, train_split_size: 20528, epochs: 2 },
      validation: { accuracy: 0.8, macro_f1: 0.55 },
      test: { accuracy: 0.7867, macro_f1: 0.6, macro_precision: 0.61, macro_recall: 0.62 },
    },
  },
}

describe('buildModelRows', () => {
  it('averages per-class precision and recall into macro values', () => {
    const lr = buildModelRows(payload, 'test').find((r) => r.key === 'logistic_regression')!
    expect(lr.macroPrecision).toBeCloseTo(0.5)
    expect(lr.macroRecall).toBeCloseTo(0.6)
    expect(lr.confusion).toHaveLength(3)
  })

  it('leaves values the backend did not report as null instead of guessing', () => {
    const svc = buildModelRows(payload, 'test').find((r) => r.key === 'linear_svc')!
    expect(svc.macroPrecision).toBeNull()
    expect(svc.weightedF1).toBeNull()
  })

  it('reports DistilBERT from its saved evaluation and says how many reviews it was trained on', () => {
    const test = buildModelRows(payload, 'test').find((r) => r.key === 'distilbert')!
    expect(test.note).toBe('trained on 6,000 reviews')
    expect(test).toMatchObject({ accuracy: 0.7867, macroF1: 0.6, macroPrecision: 0.61, macroRecall: 0.62 })
    const validation = buildModelRows(payload, 'validation').find((r) => r.key === 'distilbert')!
    expect(validation.accuracy).toBe(0.8)
  })

  it('shows no DistilBERT row when it has not been trained', () => {
    const without = { models: { logistic_regression: payload.models!.logistic_regression } }
    expect(buildModelRows(without, 'test').map((r) => r.key)).toEqual(['logistic_regression'])
  })

  it('prefers saved macro averages over recomputing them', () => {
    const lr = buildModelRows({ models: { logistic_regression: { test: { macro_precision: 0.9, negative_precision: 0.1, neutral_precision: 0.1, positive_precision: 0.1 } } } }, 'test')[0]!
    expect(lr.macroPrecision).toBe(0.9)
  })

  it('returns no rows when the payload has no models', () => {
    expect(buildModelRows(undefined, 'test')).toEqual([])
    expect(buildModelRows({}, 'test')).toEqual([])
  })

  it('picks the best model by macro F1', () => {
    expect(bestByMacroF1(buildModelRows(payload, 'test'))).toBe('logistic_regression')
    expect(bestByMacroF1([])).toBeNull()
  })
})

describe('topic helpers', () => {
  const topic = (id: number, pct: number, words: string[]): TopicItem => ({
    topic_id: id,
    label: `Topic ${id + 1}: ${words[0]}`,
    named_by: 'top words',
    top_words: words,
    sample_reviews: [],
    prevalence_count: pct * 10,
    prevalence_pct: pct,
    avg_rating: 4.2,
    sentiment_counts: { positive: 1, neutral: 1, negative: 1 },
  })

  it('strips the numeric prefix from topic labels', () => {
    expect(topicName('Topic 3: Gift & Card')).toBe('Gift & Card')
  })

  it('lists each word once and makes words in common topics bigger', () => {
    const words = buildCloudWords([topic(0, 50, ['great', 'works']), topic(1, 5, ['great', 'card'])])
    expect(words.filter((w) => w.word === 'great')).toHaveLength(1)
    const size = (word: string) => words.find((w) => w.word === word)!.size
    expect(size('great')).toBeGreaterThan(size('card'))
  })
})
