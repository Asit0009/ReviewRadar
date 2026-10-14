import { useMemo } from 'react'
import { useCategories } from '@/api/queries'
import { useFilters } from '@/context/FilterContext'
import { categoryLabel, fmtInt } from '@/lib/format'
import { Select } from './ui/controls'

const ALL = '__all__'

/** Category filter shared by every page. The chosen value is sent to the API as the `category` parameter. */
export function CategorySelect({ hideLabel = false, className }: { hideLabel?: boolean; className?: string }) {
  const { category, setCategory } = useFilters()
  const { data } = useCategories()

  const options = useMemo(() => {
    // The list comes from the API only. There is no hard-coded copy that could go stale when the dataset changes.
    const items = [...(data?.categories ?? [])]
      .sort((a, b) => a.category.localeCompare(b.category))
      .map((c) => ({ value: c.category, label: `${categoryLabel(c.category)} (${fmtInt(c.review_count)})` }))
    return [{ value: ALL, label: 'All Categories' }, ...items]
  }, [data])

  return (
    <Select
      label="Category"
      hideLabel={hideLabel}
      className={className}
      value={category ?? ALL}
      options={options}
      onChange={(value) => setCategory(value === ALL ? null : value)}
    />
  )
}
