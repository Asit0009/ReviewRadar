import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, XAxis, YAxis } from 'recharts'
import type { CategoryItem } from '@/api/types'
import { useFilters } from '@/context/FilterContext'
import { categoryLabel, fmtInt } from '@/lib/format'
import { COLORS } from '@/lib/theme'

/** Average rating per category as bars, plus a table with the other numbers. Clicking a category selects it as the filter. */
export function CategoryCompare({ categories }: { categories: CategoryItem[] }) {
  const { category, setCategory } = useFilters()
  const sorted = [...categories].sort((a, b) => b.mean_rating - a.mean_rating)
  const rows = sorted.map((c) => ({ ...c, name: categoryLabel(c.category) }))

  return (
    <div className="space-y-4">
      <div style={{ height: rows.length * 30 + 28 }} role="img" aria-label={`Average rating by category: ${rows.map((r) => `${r.name} ${r.mean_rating}`).join(', ')}`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 34, bottom: 0, left: 0 }} barSize={14}>
            <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 4" horizontal={false} />
            <XAxis type="number" domain={[0, 5]} ticks={[0, 1, 2, 3, 4, 5]} tick={{ fill: COLORS.axis, fontSize: 11 }} tickLine={false} axisLine={{ stroke: COLORS.grid }} />
            <YAxis type="category" dataKey="name" width={150} tick={{ fill: COLORS.axis, fontSize: 11 }} tickLine={false} axisLine={false} interval={0} />
            <Bar dataKey="mean_rating" radius={[0, 6, 6, 0]} isAnimationActive={false} onClick={(d) => setCategory((d as unknown as { category: string }).category)} cursor="pointer">
              {rows.map((r) => (
                <Cell key={r.category} fill={category === null || category === r.category ? COLORS.blue : '#2a3a78'} />
              ))}
              <LabelList dataKey="mean_rating" position="right" fill={COLORS.axis} fontSize={11} formatter={(v: unknown) => (typeof v === 'number' ? v.toFixed(2) : '')} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] border-separate border-spacing-0 text-[13px]">
          <caption className="sr-only">Review statistics by category</caption>
          <thead>
            <tr className="text-left text-xs text-muted">
              <th scope="col" className="px-3 py-2 font-medium">Category</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Reviews</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Products</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Avg rating</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Verified</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Avg length</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((c) => {
              const selected = category === c.category
              return (
                <tr key={c.category} className={selected ? 'bg-brand/15' : ''}>
                  <th scope="row" className="border-t border-line-soft px-3 py-2 text-left font-medium">
                    <button
                      type="button"
                      aria-pressed={selected}
                      onClick={() => setCategory(selected ? null : c.category)}
                      className="text-left text-ink hover:text-brand-2 hover:underline"
                    >
                      {categoryLabel(c.category)}
                    </button>
                  </th>
                  <td className="border-t border-line-soft px-3 py-2 text-right text-ink-2 tabular-nums">{fmtInt(c.review_count)}</td>
                  <td className="border-t border-line-soft px-3 py-2 text-right text-ink-2 tabular-nums">{fmtInt(c.unique_products)}</td>
                  <td className="border-t border-line-soft px-3 py-2 text-right text-ink-2 tabular-nums">{c.mean_rating.toFixed(2)}</td>
                  <td className="border-t border-line-soft px-3 py-2 text-right text-ink-2 tabular-nums">{c.verified_pct.toFixed(1)}%</td>
                  <td className="border-t border-line-soft px-3 py-2 text-right text-ink-2 tabular-nums">{fmtInt(Math.round(c.mean_text_length))} chars</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
