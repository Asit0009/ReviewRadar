import { useMemo } from 'react'
import { CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipContentProps } from 'recharts'
import type { TrendPoint } from '@/api/types'
import { fmtInt, fmtMonth, fmtMonthShort, fmtPct } from '@/lib/format'
import { COLORS } from '@/lib/theme'

interface Row extends TrendPoint {
  label: string
}

function TrendTooltip({ active, payload }: TooltipContentProps) {
  const row = payload?.[0]?.payload as Row | undefined
  if (!active || !row) return null
  const line = (color: string, text: string) => (
    <p className="text-ink-2">
      <span className="mr-1.5 inline-block size-2 rounded-full" style={{ background: color }} />
      {text}
    </p>
  )
  return (
    <div className="rounded-lg border border-line bg-card-2 px-3 py-2 text-xs shadow-xl">
      <p className="mb-1 font-semibold text-ink">{fmtMonth(row.year_month)}</p>
      {line(COLORS.positive, `${fmtInt(row.positive_count)} positive`)}
      {line(COLORS.neutral, `${fmtInt(row.neutral_count)} neutral`)}
      {line(COLORS.negative, `${fmtInt(row.negative_count)} negative`)}
      <p className="mt-1 border-t border-line pt-1 text-muted">
        {fmtInt(row.review_count)} reviews · {row.avg_rating.toFixed(2)} avg rating · {fmtPct(row.positive_ratio * 100, 0)} positive
      </p>
    </div>
  )
}

/** Reviews per month, split into positive / neutral / negative (sentiment comes from the star rating). */
export function TrendChart({ points, height = 224 }: { points: TrendPoint[]; height?: number }) {
  const rows: Row[] = useMemo(() => points.map((p) => ({ ...p, label: fmtMonthShort(p.year_month) })), [points])
  const first = points[0]
  const last = points[points.length - 1]

  return (
    <div
      style={{ height }}
      role="img"
      aria-label={first && last ? `Monthly positive, neutral and negative review counts from ${fmtMonth(first.year_month)} to ${fmtMonth(last.year_month)}` : 'Monthly sentiment trend'}
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
          <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 4" vertical={false} />
          <XAxis dataKey="label" tick={{ fill: COLORS.axis, fontSize: 11 }} tickLine={false} axisLine={{ stroke: COLORS.grid }} minTickGap={32} />
          <YAxis tick={{ fill: COLORS.axis, fontSize: 11 }} tickLine={false} axisLine={false} width={44} allowDecimals={false} />
          <Tooltip content={TrendTooltip} cursor={{ stroke: COLORS.axis, strokeDasharray: '3 3' }} />
          <Line type="monotone" dataKey="positive_count" stroke={COLORS.positive} strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
          <Line type="monotone" dataKey="neutral_count" stroke={COLORS.neutral} strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
          <Line type="monotone" dataKey="negative_count" stroke={COLORS.negative} strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

export function TrendLegend() {
  const items = [
    ['Positive', COLORS.positive],
    ['Neutral', COLORS.neutral],
    ['Negative', COLORS.negative],
  ] as const
  return (
    <ul className="flex items-center gap-3 text-xs text-ink-2">
      {items.map(([label, color]) => (
        <li key={label} className="flex items-center gap-1.5">
          <span aria-hidden="true" className="size-2 rounded-full" style={{ background: color }} />
          {label}
        </li>
      ))}
    </ul>
  )
}
