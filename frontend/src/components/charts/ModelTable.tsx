import { useMemo, useState } from 'react'
import { useModelMetrics } from '@/api/queries'
import { bestByMacroF1, buildModelRows, type EvalSplit, type ModelRow } from '@/lib/models'
import { fmtRatio, fmtScore } from '@/lib/format'
import { Chip, Tabs } from '../ui/controls'
import { QueryBoundary } from '../ui/StateBlocks'
import type { ModelMetricsResponse } from '@/api/types'

const NA = (
  <span title="Not reported by the backend" className="text-muted">
    n/a
  </span>
)

const percentCell = (v: number | null) => (v === null ? NA : fmtRatio(v))
const scoreCell = (v: number | null) => (v === null ? NA : fmtScore(v))

export function ModelTableView({ rows, bestKey }: { rows: ModelRow[]; bestKey: ModelRow['key'] | null }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[460px] border-separate border-spacing-0 text-[13px]">
        <thead>
          <tr className="text-left text-xs text-muted">
            <th scope="col" className="px-3 py-2 font-medium">Model</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">Accuracy</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">Macro F1</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">Macro precision</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">Macro recall</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const best = row.key === bestKey
            const cell = `border-t border-line-soft px-3 py-2.5 text-right tabular-nums ${best ? 'bg-brand/20 text-ink' : 'text-ink-2'}`
            return (
              <tr key={row.key}>
                <th scope="row" className={`border-t border-line-soft px-3 py-2.5 text-left font-medium ${best ? 'rounded-l-lg bg-brand/20 text-ink' : 'text-ink'}`}>
                  <span className="flex flex-wrap items-center gap-2">
                    {row.name}
                    {row.note && <Chip tone="violet">{row.note}</Chip>}
                  </span>
                </th>
                <td className={cell}>{percentCell(row.accuracy)}</td>
                <td className={cell}>{scoreCell(row.macroF1)}</td>
                <td className={cell}>{percentCell(row.macroPrecision)}</td>
                <td className={`${cell} ${best ? 'rounded-r-lg' : ''}`}>{percentCell(row.macroRecall)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

const SPLIT_TABS: Array<{ value: EvalSplit; label: string }> = [
  { value: 'test', label: 'Test set' },
  { value: 'validation', label: 'Validation set' },
]

/** Model comparison from GET /models/metrics. The split toggle lives next to the table so both stay in sync. */
export function useModelTable(data: ModelMetricsResponse | undefined, split: EvalSplit) {
  return useMemo(() => {
    const rows = buildModelRows(data?.metrics, split)
    return { rows, bestKey: bestByMacroF1(rows) }
  }, [data, split])
}

function ModelTableBody({ data, split }: { data: ModelMetricsResponse; split: EvalSplit }) {
  const { rows, bestKey } = useModelTable(data, split)
  return <ModelTableView rows={rows} bestKey={bestKey} />
}

export function ModelTable({ className }: { className?: string }) {
  const query = useModelMetrics()
  const [split, setSplit] = useState<EvalSplit>('test')
  return (
    <div className={className}>
      <div className="mb-2">
        <Tabs label="Evaluation split" size="sm" value={split} options={SPLIT_TABS} onChange={setSplit} />
      </div>
      <QueryBoundary query={query} isEmpty={(d) => buildModelRows(d.metrics, split).length === 0} emptyTitle="No model results yet" emptyHint="Train the models to see accuracy and F1 scores here." loadingLabel="Loading model results">
        {(data) => <ModelTableBody data={data} split={split} />}
      </QueryBoundary>
    </div>
  )
}
