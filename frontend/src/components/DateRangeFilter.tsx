import { useId } from 'react'
import { useOverview } from '@/api/queries'
import { useFilters } from '@/context/FilterContext'

const inputClass =
  'h-9 w-full min-w-0 rounded-lg border border-line bg-card-2 px-2.5 text-[13px] text-ink hover:border-brand-2/60 [&::-webkit-calendar-picker-indicator]:opacity-60'

/**
 * From / To review dates. They are sent to the API as date_from / date_to, so every number, chart and review
 * list that follows the filters is recalculated in SQL (nothing is trimmed on the client).
 */
export function DateRangeFilter({ className = '' }: { className?: string }) {
  const { dateFrom, dateTo, setDateFrom, setDateTo, clearDates, dateRangeInvalid } = useFilters()
  // The unfiltered overview tells us the first and last review dates, which bound the pickers.
  const bounds = useOverview({})
  const min = bounds.data?.date_min ?? undefined
  const max = bounds.data?.date_max ?? undefined
  const fromId = useId()
  const toId = useId()

  return (
    <div className={className}>
      <div className="grid grid-cols-2 gap-2">
        <div className="min-w-0">
          <label htmlFor={fromId} className="mb-1 block text-xs font-medium text-muted">
            From
          </label>
          <input id={fromId} type="date" className={inputClass} min={min} max={dateTo ?? max} value={dateFrom ?? ''} onChange={(e) => setDateFrom(e.target.value || null)} />
        </div>
        <div className="min-w-0">
          <label htmlFor={toId} className="mb-1 block text-xs font-medium text-muted">
            To
          </label>
          <input id={toId} type="date" className={inputClass} min={dateFrom ?? min} max={max} value={dateTo ?? ''} onChange={(e) => setDateTo(e.target.value || null)} />
        </div>
      </div>
      {dateRangeInvalid && (
        <p role="alert" className="mt-1.5 text-xs text-neg">
          The start date must be on or before the end date. The dates are ignored until they are fixed.
        </p>
      )}
      {(dateFrom || dateTo) && (
        <button type="button" onClick={clearDates} className="mt-1.5 text-xs font-medium text-brand-2 hover:underline">
          Clear dates
        </button>
      )}
    </div>
  )
}
