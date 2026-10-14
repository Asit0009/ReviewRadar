import { TriangleAlert } from 'lucide-react'
import { useCategoryMetrics } from '@/api/queries'
import type { CategoryMetricsTable as Table } from '@/api/types'
import { categoryLabel, fmtInt, fmtRatio, fmtScore } from '@/lib/format'
import { Callout } from '../ui/controls'
import { QueryBoundary } from '../ui/StateBlocks'

const MODEL_NAMES: Record<string, string> = {
  logistic_regression: 'Logistic Regression',
  linear_svc: 'Linear SVC',
  distilbert: 'DistilBERT',
}

function TableView({ data }: { data: Table }) {
  const models = Object.keys(data.models)
  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] border-separate border-spacing-0 text-[13px]">
          <caption className="sr-only">Test-set accuracy and macro F1 for each model in each category</caption>
          <thead>
            <tr className="text-left text-xs text-muted">
              <th scope="col" className="px-3 py-2 font-medium">Category</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Test reviews</th>
              {models.map((m) => (
                <th key={m} scope="col" className="px-3 py-2 text-right font-medium">
                  {MODEL_NAMES[m] ?? m}
                  <span className="block text-[10px] font-normal">accuracy · macro F1</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.categories.map((category) => {
              const first = data.models[models[0] ?? '']?.[category]
              return (
                <tr key={category}>
                  <th scope="row" className="border-t border-line-soft px-3 py-2 text-left font-medium text-ink">{categoryLabel(category)}</th>
                  <td className="border-t border-line-soft px-3 py-2 text-right text-ink-2 tabular-nums">{first ? fmtInt(first.test_samples) : 'n/a'}</td>
                  {models.map((m) => {
                    const r = data.models[m]?.[category]
                    return (
                      <td key={m} className="border-t border-line-soft px-3 py-2 text-right text-ink-2 tabular-nums">
                        {r ? (
                          <span title={r.reliable ? undefined : r.note}>
                            {fmtRatio(r.accuracy)} · {fmtScore(r.macro_f1, 2)}
                            {!r.reliable && <TriangleAlert size={12} className="ml-1 inline text-neu" aria-label="Few reviews in a class: low reliability" />}
                          </span>
                        ) : (
                          <span className="text-muted">n/a</span>
                        )}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <Callout>
        Each row scores the same saved models on only that category's test reviews. A warning triangle means at least one sentiment class has fewer than {data.min_class_support_for_reliable} test reviews in that category,
        so the score can swing a lot and should not be compared with other categories.
      </Callout>
    </div>
  )
}

export function CategoryMetricsTable() {
  const query = useCategoryMetrics()
  return (
    <QueryBoundary query={query} isEmpty={(d) => d.categories.length === 0} emptyTitle="No per-category results saved" emptyHint="Retrain the models to save them." loadingLabel="Loading per-category results" loadingRows={6}>
      {(data) => <TableView data={data} />}
    </QueryBoundary>
  )
}
