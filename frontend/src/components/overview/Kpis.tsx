import { Frown, MessageSquareText, Smile, Star, type LucideIcon } from 'lucide-react'
import { useOverview } from '@/api/queries'
import { useFilters } from '@/context/FilterContext'
import { categoryLabel, fmtInt, fmtPct } from '@/lib/format'
import { COLORS } from '@/lib/theme'
import { Card } from '../ui/Card'
import { ErrorBlock } from '../ui/StateBlocks'

interface KpiProps {
  icon: LucideIcon
  color: string
  label: string
  value: string
  hint: string
}

function Kpi({ icon: Icon, color, label, value, hint }: KpiProps) {
  return (
    <div className="flex items-center gap-3.5 rounded-xl border border-line bg-card/90 p-4">
      <span aria-hidden="true" className="flex size-11 shrink-0 items-center justify-center rounded-xl" style={{ background: `${color}2e`, color }}>
        <Icon size={22} />
      </span>
      <div className="min-w-0">
        <p className="truncate text-xs text-ink-2">{label}</p>
        <p className="mt-0.5 text-2xl leading-tight font-semibold tracking-tight text-ink tabular-nums">{value}</p>
        <p className="mt-0.5 truncate text-xs text-muted">{hint}</p>
      </div>
    </div>
  )
}

/** The four headline numbers. They follow the category filter. */
export function KpiRow() {
  const { category, scope } = useFilters()
  const { data, isError, error, refetch } = useOverview(scope)
  const scopeLabel = category ? categoryLabel(category) : 'All categories'

  if (isError) {
    return (
      <Card title="Headline numbers">
        <ErrorBlock error={error} onRetry={() => void refetch()} />
      </Card>
    )
  }

  if (!data) {
    return (
      <div className="grid grid-cols-2 gap-3 @3xl:grid-cols-4" role="status" aria-label="Loading headline numbers">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="skeleton h-[92px] rounded-xl" />
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 gap-3 @sm:grid-cols-2 @3xl:grid-cols-4">
      <Kpi icon={MessageSquareText} color={COLORS.blue} label="Total Reviews" value={fmtInt(data.total_reviews)} hint={`${fmtInt(data.total_products)} products`} />
      <Kpi icon={Star} color={COLORS.neutral} label="Average Rating" value={data.average_rating === null ? 'n/a' : `${data.average_rating.toFixed(2)} / 5`} hint={scopeLabel} />
      <Kpi icon={Smile} color={COLORS.positive} label="Positive Reviews" value={fmtPct(data.sentiment_percentages.positive)} hint={`${fmtInt(data.sentiment_distribution.positive)} reviews`} />
      <Kpi icon={Frown} color={COLORS.negative} label="Negative Reviews" value={fmtPct(data.sentiment_percentages.negative)} hint={`${fmtInt(data.sentiment_distribution.negative)} reviews`} />
    </div>
  )
}
