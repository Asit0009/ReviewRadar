import { AlertTriangle, CircleSlash, Inbox, RefreshCw, WifiOff } from 'lucide-react'
import type { ReactNode } from 'react'
import { ApiError } from '@/api/client'

/** Loading, empty, error and "not available yet" states, written once so every panel behaves the same. */

export function LoadingBlock({ label = 'Loading…', rows = 3, className = '' }: { label?: string; rows?: number; className?: string }) {
  return (
    <div role="status" aria-live="polite" className={`flex flex-col gap-2.5 py-1 ${className}`}>
      <span className="sr-only">{label}</span>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="skeleton h-4" style={{ width: `${100 - ((i * 17) % 40)}%` }} />
      ))}
    </div>
  )
}

interface MessageProps {
  icon: ReactNode
  title: string
  hint?: ReactNode
  action?: ReactNode
  tone?: 'neutral' | 'warn' | 'error'
  className?: string
}

function Message({ icon, title, hint, action, tone = 'neutral', className = '' }: MessageProps) {
  const toneClass = tone === 'error' ? 'text-neg' : tone === 'warn' ? 'text-neu' : 'text-muted'
  return (
    <div className={`flex flex-col items-center justify-center gap-1.5 px-3 py-6 text-center ${className}`}>
      <div className={toneClass} aria-hidden="true">
        {icon}
      </div>
      <p className="text-sm font-medium text-ink">{title}</p>
      {hint && <p className="max-w-sm text-xs leading-relaxed text-muted">{hint}</p>}
      {action && <div className="mt-1.5">{action}</div>}
    </div>
  )
}

export function EmptyBlock({ title = 'Nothing to show', hint, className }: { title?: string; hint?: ReactNode; className?: string }) {
  return <Message icon={<Inbox size={22} />} title={title} hint={hint} className={className} />
}

/** For things the backend cannot provide yet. Say so plainly instead of showing placeholder numbers. */
export function NotAvailableBlock({ title = 'Not available yet', hint, className }: { title?: string; hint?: ReactNode; className?: string }) {
  return <Message icon={<CircleSlash size={22} />} title={title} hint={hint} tone="warn" className={className} />
}

export function RetryButton({ onRetry }: { onRetry: () => void }) {
  return (
    <button
      type="button"
      onClick={onRetry}
      className="inline-flex items-center gap-1.5 rounded-md border border-line bg-card-2 px-2.5 py-1 text-xs font-medium text-ink-2 hover:border-brand-2 hover:text-ink"
    >
      <RefreshCw size={12} aria-hidden="true" /> Try again
    </button>
  )
}

export function ErrorBlock({ error, onRetry, className }: { error: unknown; onRetry?: () => void; className?: string }) {
  const apiError = error instanceof ApiError ? error : null
  const message = error instanceof Error ? error.message : 'Something went wrong.'
  const action = onRetry ? <RetryButton onRetry={onRetry} /> : undefined

  // 503 means the backend is running but a dataset, model or index has not been built yet.
  if (apiError?.kind === 'unavailable') {
    return <Message icon={<CircleSlash size={22} />} title="Not available yet" hint={message} tone="warn" action={action} className={className} />
  }
  if (apiError?.kind === 'network') {
    return <Message icon={<WifiOff size={22} />} title="Can't reach the API" hint={message} tone="error" action={action} className={className} />
  }
  return <Message icon={<AlertTriangle size={22} />} title="Couldn't load this data" hint={message} tone="error" action={action} className={className} />
}

interface QueryLike<T> {
  data: T | undefined
  isPending: boolean
  isError: boolean
  error: unknown
  refetch: () => unknown
}

interface QueryBoundaryProps<T> {
  query: QueryLike<T>
  children: (data: T) => ReactNode
  isEmpty?: (data: T) => boolean
  emptyTitle?: string
  emptyHint?: ReactNode
  loadingLabel?: string
  loadingRows?: number
  className?: string
}

/** Renders the right state for a query so panels only describe the happy path. */
export function QueryBoundary<T>({
  query,
  children,
  isEmpty,
  emptyTitle,
  emptyHint,
  loadingLabel,
  loadingRows,
  className,
}: QueryBoundaryProps<T>) {
  if (query.isError) return <ErrorBlock error={query.error} onRetry={() => void query.refetch()} className={className} />
  if (query.isPending || query.data === undefined) return <LoadingBlock label={loadingLabel} rows={loadingRows} className={className} />
  if (isEmpty?.(query.data)) return <EmptyBlock title={emptyTitle} hint={emptyHint} className={className} />
  return <>{children(query.data)}</>
}
