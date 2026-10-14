import { ChevronDown, Info, Star } from 'lucide-react'
import { useId, type KeyboardEvent, type ReactNode } from 'react'
import { SENTIMENT_META, isSentiment } from '@/lib/theme'

/** Small form and display controls shared across pages. */

// ---------- Select ----------
interface SelectOption<V extends string | number> {
  value: V
  label: string
}

interface SelectProps<V extends string | number> {
  label: string
  value: V
  options: SelectOption<V>[]
  onChange: (value: V) => void
  /** Hide the label visually but keep it for screen readers (e.g. when the surrounding card already names it). */
  hideLabel?: boolean
  className?: string
  disabled?: boolean
}

export function Select<V extends string | number>({ label, value, options, onChange, hideLabel, className = '', disabled }: SelectProps<V>) {
  const id = useId()
  return (
    <div className={`min-w-0 ${className}`}>
      <label htmlFor={id} className={hideLabel ? 'sr-only' : 'mb-1 block text-xs font-medium text-muted'}>
        {label}
      </label>
      <div className="relative">
        <select
          id={id}
          value={value}
          disabled={disabled}
          onChange={(e) => {
            const picked = options.find((o) => String(o.value) === e.target.value)
            if (picked) onChange(picked.value)
          }}
          className="h-9 w-full appearance-none rounded-lg border border-line bg-card-2 pr-8 pl-3 text-sm text-ink hover:border-brand-2/60 disabled:opacity-50"
        >
          {options.map((o) => (
            <option key={String(o.value)} value={o.value} className="bg-card text-ink">
              {o.label}
            </option>
          ))}
        </select>
        <ChevronDown size={15} aria-hidden="true" className="pointer-events-none absolute top-1/2 right-2.5 -translate-y-1/2 text-muted" />
      </div>
    </div>
  )
}

// ---------- Tabs ----------
interface TabOption<V extends string> {
  value: V
  label: string
  /** Optional count badge, e.g. "Positive (12)". */
  badge?: ReactNode
  disabled?: boolean
}

interface TabsProps<V extends string> {
  label: string
  value: V
  options: TabOption<V>[]
  onChange: (value: V) => void
  size?: 'sm' | 'md'
}

export function Tabs<V extends string>({ label, value, options, onChange, size = 'md' }: TabsProps<V>) {
  const enabled = options.filter((o) => !o.disabled)

  // Arrow keys move between tabs, the way a native tab list behaves.
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return
    const index = enabled.findIndex((o) => o.value === value)
    const next = enabled[(index + (e.key === 'ArrowRight' ? 1 : -1) + enabled.length) % enabled.length]
    if (next) {
      onChange(next.value)
      ;(e.currentTarget.querySelector(`[data-tab="${next.value}"]`) as HTMLElement | null)?.focus()
    }
    e.preventDefault()
  }

  return (
    <div role="tablist" aria-label={label} onKeyDown={onKeyDown} className="flex flex-wrap gap-1.5">
      {options.map((o) => {
        const selected = o.value === value
        return (
          <button
            key={o.value}
            type="button"
            role="tab"
            data-tab={o.value}
            aria-selected={selected}
            tabIndex={selected ? 0 : -1}
            disabled={o.disabled}
            onClick={() => onChange(o.value)}
            className={`inline-flex items-center gap-1.5 rounded-full border font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
              size === 'sm' ? 'px-3 py-1 text-xs' : 'px-3.5 py-1.5 text-[13px]'
            } ${
              selected
                ? 'border-brand/60 bg-brand/25 text-ink'
                : 'border-transparent bg-card-2/70 text-muted hover:text-ink'
            }`}
          >
            {o.label}
            {o.badge !== undefined && <span className="text-[11px] opacity-80">{o.badge}</span>}
          </button>
        )
      })}
    </div>
  )
}

// ---------- Rating + sentiment ----------
export function StarRating({ rating, size = 13 }: { rating: number; size?: number }) {
  const filled = Math.round(rating)
  return (
    <span role="img" aria-label={`${rating} out of 5 stars`} className="inline-flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map((n) => (
        <Star key={n} size={size} aria-hidden="true" className={n <= filled ? 'fill-neu text-neu' : 'text-line'} />
      ))}
    </span>
  )
}

export function SentimentBadge({ sentiment }: { sentiment: string }) {
  if (!isSentiment(sentiment)) {
    return <span className="rounded-full border border-line px-2 py-0.5 text-[11px] text-muted">{sentiment}</span>
  }
  const meta = SENTIMENT_META[sentiment]
  return <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${meta.chip}`}>{meta.label}</span>
}

export function Chip({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'blue' | 'teal' | 'violet' }) {
  const tones = {
    neutral: 'border-line bg-card-2/70 text-ink-2',
    blue: 'border-blue/30 bg-blue/15 text-blue',
    teal: 'border-teal/30 bg-teal/15 text-teal',
    violet: 'border-violet/30 bg-violet/15 text-violet',
  }
  return <span className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}>{children}</span>
}

// ---------- Notes ----------
export function Callout({ tone = 'info', children, className = '' }: { tone?: 'info' | 'warn'; children: ReactNode; className?: string }) {
  const styles = tone === 'warn' ? 'border-neu/30 bg-neu/10 text-ink-2' : 'border-blue/25 bg-blue/10 text-ink-2'
  return (
    <div role="note" className={`flex items-start gap-2 rounded-lg border px-3 py-2 text-xs leading-relaxed ${styles} ${className}`}>
      <Info size={14} aria-hidden="true" className={`mt-0.5 shrink-0 ${tone === 'warn' ? 'text-neu' : 'text-blue'}`} />
      <div>{children}</div>
    </div>
  )
}

/** A hover/focus hint for short explanations that should not take up space all the time. */
export function InfoTip({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex">
      <button type="button" aria-label={text} className="rounded-full text-muted hover:text-ink">
        <Info size={14} aria-hidden="true" />
      </button>
      {/*
        `hidden` (display: none) until hover/focus, so the box takes no part in layout while closed. A merely transparent
        box still widens the page. It is anchored to the icon's right edge and opens leftwards because these icons sit
        at the right end of card headers.
      */}
      <span
        role="tooltip"
        className="pointer-events-none absolute top-full right-0 z-40 mt-1.5 hidden w-56 max-w-[70vw] rounded-md border border-line bg-card-2 px-2.5 py-1.5 text-[11px] leading-snug text-ink-2 shadow-lg group-focus-within:block group-hover:block"
      >
        {text}
      </span>
    </span>
  )
}

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight text-ink">{title}</h1>
        {description && <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-muted">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-end gap-3">{actions}</div>}
    </div>
  )
}

export function ProgressBar({ value, color, label }: { value: number; color: string; label: string }) {
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuenow={Math.round(value)}
      aria-valuemin={0}
      aria-valuemax={100}
      className="h-2 w-full overflow-hidden rounded-full bg-card-2"
    >
      <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${Math.min(100, Math.max(0, value))}%`, background: color }} />
    </div>
  )
}
