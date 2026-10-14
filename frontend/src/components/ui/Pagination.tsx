import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useId, useState, type FormEvent } from 'react'
import { fmtInt } from '@/lib/format'

interface PaginationProps {
  page: number
  totalPages: number
  onPageChange: (page: number) => void
  disabled?: boolean
}

const buttonClass =
  'inline-flex h-8 items-center gap-1 rounded-lg border border-line bg-card-2 px-3 text-xs font-medium text-ink-2 hover:border-brand-2/60 hover:text-ink disabled:cursor-not-allowed disabled:opacity-40'

/** Previous / next buttons plus a "jump to page" box, since the review list can run to hundreds of pages. */
export function Pagination({ page, totalPages, onPageChange, disabled }: PaginationProps) {
  // null means "show the current page"; the box only holds its own text while the user is typing.
  const [typed, setTyped] = useState<string | null>(null)
  const id = useId()
  const shown = typed ?? String(page)

  const go = (target: number) => {
    setTyped(null)
    if (target !== page) onPageChange(target)
  }

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const n = Number.parseInt(shown, 10)
    go(Number.isNaN(n) ? page : Math.min(Math.max(n, 1), totalPages))
  }

  return (
    <nav aria-label="Pagination" className="flex flex-wrap items-center justify-between gap-3">
      <p className="text-xs text-muted" aria-live="polite">
        Page {fmtInt(page)} of {fmtInt(totalPages)}
      </p>
      <div className="flex items-center gap-2">
        <button type="button" className={buttonClass} disabled={disabled || page <= 1} onClick={() => go(page - 1)}>
          <ChevronLeft size={14} aria-hidden="true" /> Previous
        </button>
        <form onSubmit={submit} className="flex items-center gap-1.5">
          <label htmlFor={id} className="sr-only">
            Go to page
          </label>
          <input
            id={id}
            inputMode="numeric"
            value={shown}
            onChange={(e) => setTyped(e.target.value.replace(/\D/g, ''))}
            className="h-8 w-14 rounded-lg border border-line bg-card-2 text-center text-xs text-ink"
          />
          <button type="submit" className={buttonClass} disabled={disabled}>
            Go
          </button>
        </form>
        <button type="button" className={buttonClass} disabled={disabled || page >= totalPages} onClick={() => go(page + 1)}>
          Next <ChevronRight size={14} aria-hidden="true" />
        </button>
      </div>
    </nav>
  )
}
