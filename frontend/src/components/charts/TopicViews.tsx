import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import type { TopicItem } from '@/api/types'
import { fmtInt, fmtPct } from '@/lib/format'
import { TOPIC_COLORS } from '@/lib/theme'
import { StackedBarRow } from './BarRows'

/** "Topic 3: Gift & Card & Gift Card" -> "Gift & Card & Gift Card". The numbering is already implied by the order. */
export const topicName = (label: string) => label.replace(/^Topic\s+\d+:\s*/i, '')

export const topicColor = (topic: TopicItem) => TOPIC_COLORS[topic.topic_id % TOPIC_COLORS.length] ?? TOPIC_COLORS[0]

const WORDS_PER_TOPIC = 3

interface CloudWord {
  word: string
  size: number
  color: string
}

/**
 * Builds the word list for the cloud. Word size comes from two real numbers: how common the topic is
 * (prevalence_pct) and how high the word ranks inside it (the API returns top_words in rank order).
 */
export function buildCloudWords(topics: TopicItem[]): CloudWord[] {
  const seen = new Set<string>()
  const words: CloudWord[] = []
  for (const topic of [...topics].sort((a, b) => b.prevalence_pct - a.prevalence_pct)) {
    topic.top_words.slice(0, WORDS_PER_TOPIC).forEach((word, rank) => {
      if (seen.has(word)) return
      seen.add(word)
      const base = 13 + 21 * Math.sqrt(Math.min(topic.prevalence_pct, 55) / 55)
      words.push({ word, size: Math.round(base * (1 - rank * 0.1)), color: topicColor(topic) })
    })
  }
  // Alternate big and small words so the large ones are spread out instead of clumped.
  const sorted = words.sort((a, b) => b.size - a.size)
  const mixed: CloudWord[] = []
  while (sorted.length) {
    const next = mixed.length % 2 === 0 ? sorted.shift() : sorted.pop()
    if (next) mixed.push(next)
  }
  return mixed
}

/** Tag cloud of the top words in each topic. Each word links to a review search for it. */
export function TopicCloud({ topics }: { topics: TopicItem[] }) {
  const words = useMemo(() => buildCloudWords(topics), [topics])
  return (
    <ul className="flex flex-wrap content-center items-center justify-center gap-x-4 gap-y-1.5 py-1" aria-label="Top words by topic. Each word opens a review search.">
      {words.map((w) => (
        <li key={w.word} className="leading-none">
          <Link
            to={`/reviews?q=${encodeURIComponent(w.word)}`}
            title={`Search reviews for "${w.word}"`}
            className="font-semibold transition-opacity hover:opacity-70"
            style={{ fontSize: w.size, color: w.color }}
          >
            {w.word}
          </Link>
        </li>
      ))}
    </ul>
  )
}

/** Topics ranked by how many reviews fall under them. */
export function TopicBars({ topics, limit }: { topics: TopicItem[]; limit?: number }) {
  const sorted = [...topics].sort((a, b) => b.prevalence_pct - a.prevalence_pct).slice(0, limit ?? topics.length)
  const max = Math.max(...sorted.map((t) => t.prevalence_pct), 1)
  return (
    <ul className="space-y-3">
      {sorted.map((t) => (
        <StackedBarRow
          key={t.topic_id}
          label={topicName(t.label)}
          value={t.prevalence_count}
          percent={(t.prevalence_pct / max) * 100}
          color={topicColor(t)}
          valueLabel={fmtPct(t.prevalence_pct, 1)}
          title={`${fmtInt(t.prevalence_count)} reviews`}
        />
      ))}
    </ul>
  )
}
