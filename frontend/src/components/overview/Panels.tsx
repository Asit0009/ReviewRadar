import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useHealth, useModelMetrics, useMonitoring, useOverview, useRatingDistribution, useSentimentTrend, useTopics } from '@/api/queries'
import { useFilters } from '@/context/FilterContext'
import { categoryLabel, fmtDate, fmtInt, fmtMonth, fmtPct } from '@/lib/format'
import { CategorySelect } from '../CategorySelect'
import { DateRangeFilter } from '../DateRangeFilter'
import { RatingBars } from '../charts/BarRows'
import { DriftChart } from '../charts/DriftChart'
import { ModelTable } from '../charts/ModelTable'
import { SentimentDonut } from '../charts/SentimentDonut'
import { TopicBars, TopicCloud } from '../charts/TopicViews'
import { TrendChart, TrendLegend } from '../charts/TrendChart'
import { Card } from '../ui/Card'
import { Callout, InfoTip, Tabs } from '../ui/controls'
import { QueryBoundary } from '../ui/StateBlocks'

/** Describes the active filters in words, for card subtitles. */
export function useScopeLabel(): string {
  const { category, scope } = useFilters()
  const parts = [category ? categoryLabel(category) : 'All categories']
  if (scope.date_from || scope.date_to) parts.push(`${scope.date_from ?? 'start'} to ${scope.date_to ?? 'today'}`)
  return parts.join(' · ')
}

/** Category and review dates. These controls drive every panel on the dashboard through the API. */
export function FilterCard() {
  const { scope } = useFilters()
  const { data } = useOverview(scope)
  return (
    <Card className="shrink-0" bodyClassName="space-y-3 pt-3.5">
      <CategorySelect />
      <DateRangeFilter />
      <p className="flex items-center gap-2 text-xs text-ink-2" aria-live="polite">
        <span aria-hidden="true" className="size-2 rounded-full bg-pos" />
        {data ? `${fmtInt(data.total_reviews)} reviews analyzed` : 'Counting reviews…'}
      </p>
    </Card>
  )
}

export function SentimentCard() {
  const { scope } = useFilters()
  const query = useOverview(scope)
  const label = useScopeLabel()
  return (
    <Card
      title="Overall Sentiment"
      className="shrink-0"
      subtitle={label}
      actions={<InfoTip text="Sentiment labels come from star ratings: 1-2 stars is negative, 3 is neutral and 4-5 is positive." />}
    >
      <QueryBoundary query={query} loadingLabel="Loading sentiment split" loadingRows={4}>
        {(data) => <SentimentDonut overview={data} size={104} />}
      </QueryBoundary>
    </Card>
  )
}

/** If the newest review falls before the end of the last charted month, that month is partial: say so. */
export function lastMonthIsIncomplete(lastYearMonth: string, newestReviewDate: string | null | undefined): string | null {
  if (!newestReviewDate || newestReviewDate.slice(0, 7) !== lastYearMonth) return null
  const [year, month, day] = newestReviewDate.split('-').map(Number)
  if (!year || !month || !day) return null
  const daysInMonth = new Date(year, month, 0).getDate()
  return day < daysInMonth ? newestReviewDate : null
}

export function TrendPanel() {
  const { scope } = useFilters()
  const query = useSentimentTrend(scope)
  const overview = useOverview(scope)
  const label = useScopeLabel()
  return (
    <Card
      title="Sentiment Over Time"
      subtitle={label}
      className="h-full"
      actions={
        <>
          <TrendLegend />
          <InfoTip text="Reviews per month by sentiment. Sentiment comes from the star rating. Months with very few reviews can swing a lot." />
        </>
      }
    >
      <QueryBoundary query={query} isEmpty={(d) => d.trend.length === 0} emptyTitle="No reviews in this period" emptyHint="Try a wider date range or another category." loadingLabel="Loading monthly trend" loadingRows={6}>
        {(data) => {
          const first = data.trend[0]
          const last = data.trend[data.trend.length - 1]
          const incompleteEnd = last ? lastMonthIsIncomplete(last.year_month, overview.data?.date_max) : null
          return (
            <>
              <TrendChart points={data.trend} height={196} />
              {first && last && (
                <p className="mt-1.5 text-[11px] text-muted">
                  {fmtMonth(first.year_month)} to {fmtMonth(last.year_month)} · {data.total_data_points} months
                  {incompleteEnd && ` · ${fmtMonth(last.year_month)} is incomplete: the data ends on ${fmtDate(incompleteEnd)}`}
                </p>
              )}
            </>
          )
        }}
      </QueryBoundary>
    </Card>
  )
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-line-soft bg-bg/40 px-3 py-2.5">
      <p className="text-[11px] text-muted">{label}</p>
      <p className="mt-0.5 text-lg font-semibold text-ink tabular-nums">{value}</p>
      {hint && <p className="text-[11px] text-muted">{hint}</p>}
    </div>
  )
}

export function RatingPanel() {
  const { scope } = useFilters()
  const query = useRatingDistribution(scope)
  return (
    <Card title="Rating Distribution" subtitle="Share of reviews by star rating" className="h-full">
      <QueryBoundary query={query} isEmpty={(d) => d.total_reviews === 0} emptyTitle="No ratings match these filters" loadingLabel="Loading rating distribution" loadingRows={5}>
        {(data) => {
          const highShare = data.buckets.filter((b) => b.stars >= 4).reduce((sum, b) => sum + b.pct, 0)
          return (
            <div className="flex h-full flex-col justify-between gap-5">
              <RatingBars buckets={data.buckets} />
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Rated 4 or 5 stars" value={fmtPct(highShare)} />
                <Stat label="Ratings counted" value={fmtInt(data.total_reviews)} />
              </div>
            </div>
          )
        }}
      </QueryBoundary>
    </Card>
  )
}

type TopicView = 'common' | 'complaints'

export function TopicsPanel() {
  const { scope } = useFilters()
  const [view, setView] = useState<TopicView>('complaints')
  const all = useTopics(scope)
  const complaints = useTopics({ ...scope, sentiment: 'negative' })
  const active = view === 'complaints' ? complaints : all

  return (
    <Card
      title="Review Topics (NLP)"
      subtitle="Found with NMF; each review is tagged with its main topic"
      className="h-full"
      actions={
        <Link to="/topics" className="text-xs font-medium text-brand-2 hover:underline">
          All topics
        </Link>
      }
    >
      <QueryBoundary query={all} isEmpty={(d) => d.topics.length === 0} emptyTitle="No topics yet" emptyHint="Run the topic model to see themes here." loadingLabel="Loading topics" loadingRows={5}>
        {(data) => (
          <div className="space-y-4">
            <TopicCloud topics={data.topics} />
            <div>
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <Tabs
                  label="Topic view"
                  size="sm"
                  value={view}
                  onChange={setView}
                  options={[
                    { value: 'complaints', label: 'Top complaints' },
                    { value: 'common', label: 'Most common' },
                  ]}
                />
              </div>
              <QueryBoundary query={active} isEmpty={(d) => d.reviews_considered === 0} emptyTitle="No tagged reviews for these filters" loadingLabel="Loading topic counts" loadingRows={3}>
                {(d) => (
                  <>
                    <p className="mb-2 text-[11px] text-muted">
                      {view === 'complaints' ? `Share of the ${fmtInt(d.reviews_considered)} negative (1-2 star) reviews` : `Share of ${fmtInt(d.reviews_considered)} tagged reviews`}
                    </p>
                    <TopicBars topics={d.topics.filter((t) => t.prevalence_count > 0)} limit={4} />
                  </>
                )}
              </QueryBoundary>
            </div>
          </div>
        )}
      </QueryBoundary>
    </Card>
  )
}

export function ModelsPanel() {
  const metrics = useModelMetrics()
  const dist = metrics.data?.metrics.split_info?.class_distribution_test
  const testTotal = dist ? dist.positive + dist.neutral + dist.negative : 0
  const neutralShare = dist && testTotal > 0 ? fmtPct((dist.neutral * 100) / testTotal, 0) : null
  return (
    <Card
      title="Model Performance"
      subtitle="Sentiment classification (ML + deep learning)"
      className="h-full"
      actions={
        <Link to="/predictions" className="text-xs font-medium text-brand-2 hover:underline">
          Try a prediction
        </Link>
      }
    >
      <ModelTable />
      <Callout className="mt-3">
        Macro F1 is lower than accuracy because neutral reviews are rare{neutralShare ? ` (${neutralShare} of the test set)` : ''} and hard to predict.
      </Callout>
    </Card>
  )
}

/** Drift and latency snapshot for the overview. The full table lives on Deploy & Monitor. */
export function MonitoringPanel() {
  const monitoring = useMonitoring()
  const health = useHealth()
  const healthy = health.data?.health.status === 'healthy'

  return (
    <Card
      title="Model Monitoring"
      subtitle="Data drift between the reference and current rows"
      className="h-full"
      actions={
        health.data && (
          <span className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${healthy ? 'border-pos/30 bg-pos/15 text-pos' : 'border-neu/30 bg-neu/15 text-neu'}`}>
            API {health.data.health.status}
          </span>
        )
      }
    >
      <QueryBoundary query={monitoring} loadingLabel="Loading drift check" loadingRows={5}>
        {(data) => (
          <div className="space-y-3">
            <div>
              <h3 className="mb-1 text-xs font-medium text-muted">Feature drift (PSI, lower is more stable)</h3>
              <DriftChart features={data.feature_metrics} />
            </div>
            <div className="grid grid-cols-3 gap-2">
              <Stat label="Drifted features" value={`${data.drifted_features_count} of ${data.total_features_evaluated}`} hint={data.status === 'healthy' ? 'Stable' : 'Check details'} />
              <Stat label="Rows compared" value={`${fmtInt(data.reference_sample_size)}`} hint={`vs ${fmtInt(data.current_sample_size)} current`} />
              <Stat label="API latency" value={health.data ? `${health.data.latencyMs} ms` : '…'} hint="Measured in your browser" />
            </div>
            <Link to="/monitoring" className="inline-block text-xs font-medium text-brand-2 hover:underline">
              Open Deploy &amp; Monitor
            </Link>
          </div>
        )}
      </QueryBoundary>
    </Card>
  )
}
