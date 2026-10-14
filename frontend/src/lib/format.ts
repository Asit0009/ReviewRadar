const intFormat = new Intl.NumberFormat('en-US')

export const fmtInt = (n: number) => intFormat.format(n)

export const fmtPct = (n: number, digits = 1) => `${n.toFixed(digits)}%`

/** 0.8396 -> "84.0%" */
export const fmtRatio = (r: number, digits = 1) => `${(r * 100).toFixed(digits)}%`

export const fmtScore = (n: number, digits = 3) => n.toFixed(digits)

/** "2023-03-13" -> "Mar 13, 2023". Parsed as UTC so the day never shifts with the viewer's timezone. */
export function fmtDate(iso: string): string {
  const d = new Date(/^\d{4}-\d{2}-\d{2}$/.test(iso) ? `${iso}T00:00:00Z` : iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' })
}

/** "2023-03" -> "Mar 2023" */
export function fmtMonth(yearMonth: string): string {
  const d = new Date(`${yearMonth}-01T00:00:00Z`)
  if (Number.isNaN(d.getTime())) return yearMonth
  return d.toLocaleDateString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' })
}

/** "2023-03" -> "Mar '23", short enough for chart axes. */
export function fmtMonthShort(yearMonth: string): string {
  const d = new Date(`${yearMonth}-01T00:00:00Z`)
  if (Number.isNaN(d.getTime())) return yearMonth
  const month = d.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' })
  return `${month} '${String(d.getUTCFullYear()).slice(2)}`
}

/** "Arts_Crafts_and_Sewing" -> "Arts Crafts and Sewing" */
export const categoryLabel = (category: string) => category.replace(/_/g, ' ')

export const titleCase = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

export function fmtTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
}
