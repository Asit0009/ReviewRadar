import { fmtInt, fmtPct } from '@/lib/format'
import type { RatingBucket } from '@/api/types'
import { COLORS } from '@/lib/theme'

interface BarRowProps {
  label: string
  value: number
  /** 0-100 width of the bar. */
  percent: number
  color: string
  valueLabel: string
  title?: string
}

/** The bar itself. Very small non-zero values still get a sliver so they stay visible. */
function Meter({ label, value, percent, color, valueLabel }: Omit<BarRowProps, 'title'>) {
  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(percent)}
      aria-valuetext={valueLabel}
      className="h-2 overflow-hidden rounded-full bg-card-2"
    >
      <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${Math.max(percent, value > 0 ? 1.5 : 0)}%`, background: color }} />
    </div>
  )
}

/** One horizontal bar with the label on the left and the value on the right (same look as the reference panels). */
export function BarRow({ title, ...bar }: BarRowProps) {
  return (
    <li className="grid grid-cols-[88px_1fr_auto] items-center gap-3 text-[13px] sm:grid-cols-[110px_1fr_auto]" title={title}>
      <span className="truncate text-ink-2">{bar.label}</span>
      <Meter {...bar} />
      <span className="w-12 text-right font-medium text-ink tabular-nums">{bar.valueLabel}</span>
    </li>
  )
}

/** Same data as BarRow, but the label sits above the bar so long names (topic titles) are not cut off. */
export function StackedBarRow({ title, ...bar }: BarRowProps) {
  return (
    <li title={title}>
      <div className="mb-1 flex items-baseline justify-between gap-3 text-[13px]">
        <span className="min-w-0 truncate text-ink-2" title={bar.label}>
          {bar.label}
        </span>
        <span className="shrink-0 font-medium text-ink tabular-nums">{bar.valueLabel}</span>
      </div>
      <Meter {...bar} />
    </li>
  )
}

const STAR_COLORS: Record<number, string> = {
  5: COLORS.positive,
  4: COLORS.positive,
  3: COLORS.neutral,
  2: COLORS.negative,
  1: COLORS.negative,
}

/** 5-star to 1-star breakdown. Bars are scaled to the largest bucket so small groups stay visible. */
export function RatingBars({ buckets }: { buckets: RatingBucket[] }) {
  const max = Math.max(...buckets.map((b) => b.count), 1)
  return (
    <ul className="space-y-3">
      {buckets.map((b) => (
        <BarRow
          key={b.stars}
          label={`${b.stars} ${b.stars === 1 ? 'star' : 'stars'}`}
          value={b.count}
          percent={(b.count / max) * 100}
          color={STAR_COLORS[b.stars] ?? COLORS.blue}
          valueLabel={fmtPct(b.pct, 1)}
          title={`${fmtInt(b.count)} reviews`}
        />
      ))}
    </ul>
  )
}
