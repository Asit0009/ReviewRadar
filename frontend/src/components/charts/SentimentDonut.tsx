import { Cell, Pie, PieChart, ResponsiveContainer } from 'recharts'
import type { OverviewResponse } from '@/api/types'
import { fmtInt, fmtPct } from '@/lib/format'
import { SENTIMENT_META, SENTIMENTS } from '@/lib/theme'

/** Donut of positive / neutral / negative shares, with the headline percentage in the middle. */
export function SentimentDonut({ overview, size = 112 }: { overview: OverviewResponse; size?: number }) {
  const counts = overview.sentiment_distribution
  const pct = overview.sentiment_percentages
  const data = SENTIMENTS.map((s) => ({ key: s, name: SENTIMENT_META[s].label, value: counts[s], color: SENTIMENT_META[s].color }))
  const summary = data.map((d) => `${d.name} ${fmtPct(pct[d.key])}`).join(', ')

  if (overview.total_reviews === 0) {
    return <p className="py-6 text-center text-xs text-muted">No reviews match these filters.</p>
  }

  return (
    <div className="flex items-center gap-4">
      <div className="relative shrink-0" style={{ width: size, height: size }} role="img" aria-label={`Sentiment split: ${summary}`}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={data} dataKey="value" innerRadius="72%" outerRadius="100%" paddingAngle={2} stroke="none" startAngle={90} endAngle={-270} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.key} fill={d.color} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-lg leading-none font-semibold text-ink">{fmtPct(pct.positive, 1)}</span>
          <span className="mt-1 text-[11px] text-muted">Positive</span>
        </div>
      </div>
      <ul className="min-w-0 flex-1 space-y-2 text-[13px]">
        {data.map((d) => (
          <li key={d.key} className="flex items-center gap-2">
            <span aria-hidden="true" className="size-2.5 shrink-0 rounded-full" style={{ background: d.color }} />
            <span className="text-ink-2">{d.name}</span>
            <span className="ml-auto font-medium text-ink tabular-nums" title={`${fmtInt(d.value)} reviews`}>
              {fmtPct(pct[d.key], 1)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}
