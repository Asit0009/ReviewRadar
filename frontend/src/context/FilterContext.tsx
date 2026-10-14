import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'
import type { ScopeParams } from '@/api/types'

interface FilterState {
  /** Raw category name as the API expects it (e.g. "Electronics"), or null for all categories. */
  category: string | null
  setCategory: (category: string | null) => void
  /** Inclusive review-date range (YYYY-MM-DD). Both are sent to the API and applied in SQL. */
  dateFrom: string | null
  dateTo: string | null
  setDateFrom: (date: string | null) => void
  setDateTo: (date: string | null) => void
  clearDates: () => void
  /** True when the start date is after the end date. The API would answer 422, so the dates are not sent. */
  dateRangeInvalid: boolean
  /** The filters in the shape every query hook sends to the API. */
  scope: ScopeParams
}

const FilterContext = createContext<FilterState | null>(null)

export function FilterProvider({ children }: { children: ReactNode }) {
  const [category, setCategory] = useState<string | null>(null)
  const [dateFrom, setDateFrom] = useState<string | null>(null)
  const [dateTo, setDateTo] = useState<string | null>(null)

  const value = useMemo<FilterState>(() => {
    const dateRangeInvalid = Boolean(dateFrom && dateTo && dateFrom > dateTo)
    return {
      category,
      setCategory,
      dateFrom,
      dateTo,
      setDateFrom,
      setDateTo,
      clearDates: () => {
        setDateFrom(null)
        setDateTo(null)
      },
      dateRangeInvalid,
      scope: { category, date_from: dateRangeInvalid ? null : dateFrom, date_to: dateRangeInvalid ? null : dateTo },
    }
  }, [category, dateFrom, dateTo])

  return <FilterContext.Provider value={value}>{children}</FilterContext.Provider>
}

export function useFilters(): FilterState {
  const ctx = useContext(FilterContext)
  if (!ctx) throw new Error('useFilters must be used inside <FilterProvider>')
  return ctx
}
