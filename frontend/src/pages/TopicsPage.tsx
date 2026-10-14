import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { Sentiment } from '@/api/types'
import { useTopics } from '@/api/queries'
import { CategorySelect } from '@/components/CategorySelect'
import { DateRangeFilter } from '@/components/DateRangeFilter'
import { TopicBars, TopicCloud, topicColor, topicName } from '@/components/charts/TopicViews'
import { PageLayout } from '@/components/layout/PageLayout'
import { useScopeLabel } from '@/components/overview/Panels'
import { Card } from '@/components/ui/Card'
import { Callout, Chip, PageHeader, ProgressBar, Select } from '@/components/ui/controls'
import { QueryBoundary } from '@/components/ui/StateBlocks'
import { useFilters } from '@/context/FilterContext'
import { fmtInt, fmtPct } from '@/lib/format'
import { cleanReviewText } from '@/lib/text'
import { SENTIMENT_META, SENTIMENTS } from '@/lib/theme'
import { usePageTitle } from '@/lib/usePageTitle'

const SENTIMENT_OPTIONS = [
  { value: '', label: 'All reviews' },
  { value: 'negative', label: 'Negative reviews (complaints)' },
  { value: 'neutral', label: 'Neutral reviews' },
  { value: 'positive', label: 'Positive reviews' },
]

export default function TopicsPage() {
  usePageTitle('Topic Modeling')
  const { scope } = useFilters()
  const [sentiment, setSentiment] = useState<Sentiment | ''>('')
  const query = useTopics({ ...scope, sentiment: sentiment || null })
  const label = useScopeLabel()

  return (
    <PageLayout>
      <PageHeader
        title="Topic Modeling"
        description="Groups of words that often appear together in review text, found with TF-IDF and NMF (non-negative matrix factorization). Every review is tagged with its strongest topic."
        actions={
          <>
            <CategorySelect className="w-52" />
            <Select label="Show topics of" className="w-56" value={sentiment} options={SENTIMENT_OPTIONS} onChange={(v) => setSentiment(v as Sentiment | '')} />
            <DateRangeFilter className="w-64" />
          </>
        }
      />
      <Callout>
        The topic model is trained once on all reviews, so topic definitions are the same everywhere. The counts, shares and example reviews below follow your filters.{' '}
        {query.data && query.data.topics.some((t) => t.named_by !== 'reviewed')
          ? "Some names are made automatically from a topic's top words, so they can read oddly."
          : "The names were chosen by reading each topic's reviews. A topic is a group of words that appear together, so it can mix two ideas."}
      </Callout>

      <QueryBoundary
        query={query}
        isEmpty={(d) => d.topics.length === 0}
        emptyTitle="No topics yet"
        emptyHint="Run the topic model first: python -m reviewpulse.nlp.topic_model"
        loadingLabel="Loading topics"
        loadingRows={6}
      >
        {(data) => (
          <>
            <div className="grid gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
              <Card title="Top words" subtitle="Bigger words belong to more common topics. Click one to search reviews." bodyClassName="flex flex-1 items-center">
                <TopicCloud topics={data.topics} />
              </Card>
              <Card title="How common each topic is" subtitle={`${label} · ${fmtInt(data.reviews_considered)} tagged reviews`}>
                {data.reviews_considered === 0 ? <p className="py-6 text-center text-xs text-muted">No tagged reviews match these filters.</p> : <TopicBars topics={data.topics.filter((t) => t.prevalence_count > 0)} />}
              </Card>
            </div>

            <div className="grid gap-4 @2xl:grid-cols-2">
              {data.topics.map((topic) => (
                <Card
                  key={topic.topic_id}
                  title={
                    <span className="flex items-center gap-2">
                      <span aria-hidden="true" className="size-2.5 rounded-full" style={{ background: topicColor(topic) }} />
                      {topicName(topic.label)}
                    </span>
                  }
                  subtitle={`${fmtInt(topic.prevalence_count)} reviews (${fmtPct(topic.prevalence_pct)})${topic.avg_rating !== null ? ` · ${topic.avg_rating.toFixed(2)} average rating` : ''}`}
                  actions={
                    <Link to={`/reviews?topic=${topic.topic_id}`} className="text-xs font-medium text-brand-2 hover:underline">
                      View reviews
                    </Link>
                  }
                >
                  <ProgressBar value={topic.prevalence_pct} color={topicColor(topic)} label={`${topicName(topic.label)} share of reviews`} />
                  <ul className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted">
                    {SENTIMENTS.map((s) => (
                      <li key={s} className="flex items-center gap-1">
                        <span aria-hidden="true" className="size-2 rounded-full" style={{ background: SENTIMENT_META[s].color }} />
                        {fmtInt(topic.sentiment_counts[s])} {s}
                      </li>
                    ))}
                  </ul>
                  <h3 className="mt-3 mb-1.5 text-xs font-medium text-muted">Top words</h3>
                  <ul className="flex flex-wrap gap-1.5">
                    {topic.top_words.map((w) => (
                      <li key={w}>
                        <Link to={`/reviews?q=${encodeURIComponent(w)}`} title={`Search reviews for "${w}"`}>
                          <Chip tone="blue">{w}</Chip>
                        </Link>
                      </li>
                    ))}
                  </ul>
                  {topic.sample_reviews.length > 0 && (
                    <>
                      <h3 className="mt-3 mb-1.5 text-xs font-medium text-muted">Example reviews</h3>
                      <ul className="space-y-1.5">
                        {topic.sample_reviews.map((s, i) => (
                          <li key={i} className="rounded-lg border border-line-soft bg-bg/40 px-3 py-2 text-xs leading-relaxed text-ink-2">
                            {cleanReviewText(s)}
                          </li>
                        ))}
                      </ul>
                    </>
                  )}
                </Card>
              ))}
            </div>
          </>
        )}
      </QueryBoundary>
    </PageLayout>
  )
}
