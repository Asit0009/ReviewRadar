import { Search, X } from 'lucide-react'
import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { ReviewSort, Sentiment, SortOrder } from '@/api/types'
import { useReviews, useTopics } from '@/api/queries'
import { CategorySelect } from '@/components/CategorySelect'
import { DateRangeFilter } from '@/components/DateRangeFilter'
import { topicName } from '@/components/charts/TopicViews'
import { PageLayout } from '@/components/layout/PageLayout'
import { ReviewCard } from '@/components/reviews/ReviewCard'
import { Card } from '@/components/ui/Card'
import { PageHeader, Select } from '@/components/ui/controls'
import { Pagination } from '@/components/ui/Pagination'
import { EmptyBlock, ErrorBlock, LoadingBlock } from '@/components/ui/StateBlocks'
import { useFilters } from '@/context/FilterContext'
import { fmtInt } from '@/lib/format'
import { usePageTitle } from '@/lib/usePageTitle'

const PAGE_SIZES = [10, 20, 50]
const SENTIMENT_OPTIONS = [
  { value: '', label: 'All sentiments' },
  { value: 'positive', label: 'Positive' },
  { value: 'neutral', label: 'Neutral' },
  { value: 'negative', label: 'Negative' },
]
const RATING_OPTIONS = [{ value: '', label: 'Any' }, ...[1, 2, 3, 4, 5].map((n) => ({ value: String(n), label: `${n} ${n === 1 ? 'star' : 'stars'}` }))]
const SORT_OPTIONS = [
  { value: 'date-desc', label: 'Newest first' },
  { value: 'date-asc', label: 'Oldest first' },
  { value: 'rating-desc', label: 'Highest rating' },
  { value: 'rating-asc', label: 'Lowest rating' },
  { value: 'helpful-desc', label: 'Most helpful' },
]

const toInt = (v: string | null, fallback: number) => {
  const n = Number.parseInt(v ?? '', 10)
  return Number.isNaN(n) || n < 1 ? fallback : n
}
const toRating = (v: string | null) => {
  const n = Number.parseInt(v ?? '', 10)
  return n >= 1 && n <= 5 ? n : null
}
const toSentiment = (v: string | null): Sentiment | null => (v === 'positive' || v === 'neutral' || v === 'negative' ? v : null)
const toTopic = (v: string | null) => {
  const n = Number.parseInt(v ?? '', 10)
  return Number.isNaN(n) || n < 0 ? null : n
}
function toSort(v: string | null): { sort: ReviewSort; order: SortOrder } {
  const [sort, order] = (v ?? 'date-desc').split('-')
  const okSort = sort === 'date' || sort === 'rating' || sort === 'helpful'
  return { sort: okSort ? sort : 'date', order: order === 'asc' ? 'asc' : 'desc' }
}

/**
 * Browse and filter every review. Text, sentiment, rating, topic, verified and sorting live in the URL
 * (?q=battery&sentiment=negative&sort=rating-asc&page=2) so the top-bar search, topic tags and "View all" links can
 * land here with filters already set. Category and dates come from the shared filters. All of them are sent to the
 * API and applied in SQL.
 */
export default function ReviewsPage() {
  usePageTitle('Reviews Analysis')
  const [params, setParams] = useSearchParams()
  const { scope, category, setCategory, dateFrom, dateTo, clearDates } = useFilters()
  const searchId = useId()
  const topicsQuery = useTopics({})

  const q = params.get('q') ?? ''
  const sentiment = toSentiment(params.get('sentiment'))
  const minRating = toRating(params.get('min'))
  const maxRating = toRating(params.get('max'))
  const topic = toTopic(params.get('topic'))
  const verifiedOnly = params.get('verified') === '1'
  const { sort, order } = toSort(params.get('sort'))
  const page = toInt(params.get('page'), 1)
  const limit = PAGE_SIZES.includes(toInt(params.get('limit'), 20)) ? toInt(params.get('limit'), 20) : 20

  const ratingRangeInvalid = minRating !== null && maxRating !== null && minRating > maxRating

  const update = (changes: Record<string, string | null>, options?: { keepPage?: boolean; replace?: boolean }) => {
    const next = new URLSearchParams(params)
    for (const [key, value] of Object.entries(changes)) {
      if (value === null || value === '') next.delete(key)
      else next.set(key, value)
    }
    if (!options?.keepPage) next.delete('page')
    setParams(next, { replace: options?.replace })
  }

  // The text box keeps its own value so typing feels instant; the URL (and the request) follow 350 ms later.
  const [draft, setDraft] = useState(q)
  const lastPushed = useRef(q)
  useEffect(() => {
    if (q !== lastPushed.current) {
      // The query changed from outside this box (top-bar search, topic word, back button).
      lastPushed.current = q
      setDraft(q)
    }
  }, [q])
  useEffect(() => {
    const trimmed = draft.trim()
    if (trimmed === q) return
    const timer = window.setTimeout(() => {
      lastPushed.current = trimmed
      const next = new URLSearchParams(params)
      if (trimmed) next.set('q', trimmed)
      else next.delete('q')
      next.delete('page')
      setParams(next, { replace: true })
    }, 350)
    return () => window.clearTimeout(timer)
  }, [draft, q, params, setParams])

  const query = useReviews(
    {
      ...scope,
      page,
      limit,
      sentiment,
      min_rating: minRating,
      max_rating: maxRating,
      search: q,
      verified_only: verifiedOnly || null,
      topic_id: topic,
      sort,
      order,
    },
    !ratingRangeInvalid,
  )

  const topicOptions = useMemo(() => {
    const topics = topicsQuery.data?.topics ?? []
    return [{ value: '', label: 'Any topic' }, ...[...topics].sort((a, b) => a.topic_id - b.topic_id).map((t) => ({ value: String(t.topic_id), label: topicName(t.label) }))]
  }, [topicsQuery.data])

  const hasFilters = Boolean(q || sentiment || minRating || maxRating || topic !== null || verifiedOnly || category || dateFrom || dateTo)
  const clearFilters = () => {
    setDraft('')
    lastPushed.current = ''
    setCategory(null)
    clearDates()
    setParams(new URLSearchParams(), { replace: true })
  }

  const data = query.data
  const totalPages = data?.total_pages ?? 1

  return (
    <PageLayout>
      <PageHeader title="Reviews Analysis" description="Browse the real review text. Filter by category, dates, sentiment, star rating, topic or a keyword." />

      <Card bodyClassName="pt-4">
        <form role="search" aria-label="Filter reviews" onSubmit={(e) => e.preventDefault()} className="grid gap-3 @lg:grid-cols-2 @3xl:grid-cols-6">
          <div className="@lg:col-span-2 @3xl:col-span-6">
            <label htmlFor={searchId} className="mb-1 block text-xs font-medium text-muted">
              Search review text
            </label>
            <div className="relative">
              <Search size={15} aria-hidden="true" className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-muted" />
              <input
                id={searchId}
                type="search"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder="e.g. battery, refund, easy to use"
                className="h-9 w-full rounded-lg border border-line bg-card-2 pr-9 pl-9 text-sm text-ink placeholder:text-muted hover:border-brand-2/60"
              />
              {draft && (
                <button type="button" onClick={() => setDraft('')} aria-label="Clear search text" className="absolute top-1/2 right-2 -translate-y-1/2 rounded p-1 text-muted hover:text-ink">
                  <X size={14} aria-hidden="true" />
                </button>
              )}
            </div>
          </div>
          <CategorySelect className="@3xl:col-span-2" />
          <DateRangeFilter className="@lg:col-span-2 @3xl:col-span-2" />
          <Select label="Sentiment" className="@3xl:col-span-2" value={sentiment ?? ''} options={SENTIMENT_OPTIONS} onChange={(v) => update({ sentiment: v })} />
          <Select label="Minimum rating" className="@3xl:col-span-2" value={minRating ? String(minRating) : ''} options={RATING_OPTIONS} onChange={(v) => update({ min: v })} />
          <Select label="Maximum rating" className="@3xl:col-span-2" value={maxRating ? String(maxRating) : ''} options={RATING_OPTIONS} onChange={(v) => update({ max: v })} />
          <Select label="Topic" className="@3xl:col-span-2" value={topic !== null ? String(topic) : ''} options={topicOptions} onChange={(v) => update({ topic: v })} />
          <Select label="Sort by" className="@3xl:col-span-2" value={`${sort}-${order}`} options={SORT_OPTIONS} onChange={(v) => update({ sort: v === 'date-desc' ? null : v })} />
          <Select label="Per page" className="@3xl:col-span-1" value={String(limit)} options={PAGE_SIZES.map((n) => ({ value: String(n), label: String(n) }))} onChange={(v) => update({ limit: v === '20' ? null : v })} />
          <label className="flex items-end gap-2 pb-2 text-[13px] text-ink-2 @3xl:col-span-1">
            <input type="checkbox" checked={verifiedOnly} onChange={(e) => update({ verified: e.target.checked ? '1' : null })} className="size-4 accent-[var(--color-brand-2)]" />
            Verified only
          </label>
        </form>
        {ratingRangeInvalid && (
          <p role="alert" className="mt-3 text-xs text-neg">
            The minimum rating cannot be higher than the maximum rating.
          </p>
        )}
      </Card>

      <div className="flex flex-wrap items-center justify-between gap-2" aria-live="polite">
        <p className="text-[13px] text-ink-2">
          {data ? (
            <>
              <strong className="font-semibold text-ink tabular-nums">{fmtInt(data.total_records)}</strong> {data.total_records === 1 ? 'review matches' : 'reviews match'}
            </>
          ) : (
            'Loading reviews…'
          )}
        </p>
        {hasFilters && (
          <button type="button" onClick={clearFilters} className="text-xs font-medium text-brand-2 hover:underline">
            Clear all filters
          </button>
        )}
      </div>

      {ratingRangeInvalid ? (
        <EmptyBlock title="Fix the rating range to see results" />
      ) : query.isError ? (
        <ErrorBlock error={query.error} onRetry={() => void query.refetch()} />
      ) : !data ? (
        <LoadingBlock label="Loading reviews" rows={6} />
      ) : data.reviews.length === 0 ? (
        <EmptyBlock title="No reviews match these filters" hint="Try a shorter keyword, a wider date or rating range, or clear the filters." />
      ) : (
        <>
          <ul className={`space-y-2.5 transition-opacity ${query.isPlaceholderData ? 'opacity-60' : ''}`}>
            {data.reviews.map((r) => (
              <li key={r.review_id}>
                <ReviewCard review={r} highlight={q} />
              </li>
            ))}
          </ul>
          <Pagination page={Math.min(data.page, totalPages)} totalPages={totalPages} disabled={query.isPlaceholderData} onPageChange={(p) => update({ page: String(p) }, { keepPage: true })} />
        </>
      )}
    </PageLayout>
  )
}
