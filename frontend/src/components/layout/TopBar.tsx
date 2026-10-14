import { Menu, Search, Sparkles } from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useHealth } from '@/api/queries'
import { useUi } from '@/context/UiContext'
import { NotificationsButton } from './Notifications'

function ApiStatus() {
  const { data, isError, isPending } = useHealth()
  let label = 'Checking API…'
  let dot = 'bg-muted'
  if (isError) {
    label = 'API offline'
    dot = 'bg-neg'
  } else if (data) {
    const healthy = data.health.status === 'healthy'
    label = healthy ? 'API healthy' : 'API degraded'
    dot = healthy ? 'bg-pos' : 'bg-neu'
  }
  return (
    <Link
      to="/monitoring"
      title={data && !isError ? `Round trip from this browser: ${data.latencyMs} ms. Open Deploy & Monitor for details.` : 'Open Deploy & Monitor'}
      className="hidden items-center gap-2 rounded-lg border border-line bg-card/70 px-3 py-1.5 text-xs text-ink-2 hover:border-brand-2/60 hover:text-ink sm:flex"
    >
      <span aria-hidden="true" className={`size-2 rounded-full ${dot} ${isPending ? 'animate-pulse' : ''}`} />
      <span>{label}</span>
      {/* A failed refetch keeps the last good data around, so only show latency while the API is answering. */}
      {data && !isError && <span className="hidden text-muted md:inline">· {data.latencyMs} ms</span>}
    </Link>
  )
}

export function TopBar() {
  const navigate = useNavigate()
  const { setNavOpen, setAssistantOpen } = useUi()
  const [query, setQuery] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  // Press "/" anywhere to jump to the search box, unless the user is already typing somewhere.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      const typing = target?.matches('input, textarea, select, [contenteditable="true"]')
      if (e.key === '/' && !typing && !e.metaKey && !e.ctrlKey) {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const q = query.trim()
    navigate(q ? `/reviews?q=${encodeURIComponent(q)}` : '/reviews')
  }

  return (
    <header className="sticky top-0 z-30 flex h-[var(--topbar-h)] items-center gap-3 border-b border-line/70 bg-bg/80 px-4 backdrop-blur-md sm:px-5">
      <button type="button" onClick={() => setNavOpen(true)} aria-label="Open menu" className="rounded-lg p-2 text-ink-2 hover:bg-card-2 lg:hidden">
        <Menu size={20} aria-hidden="true" />
      </button>

      <form role="search" onSubmit={submit} className="relative min-w-0 max-w-[600px] flex-1">
        <label htmlFor="global-search" className="sr-only">
          Search reviews
        </label>
        <Search size={16} aria-hidden="true" className="pointer-events-none absolute top-1/2 left-3.5 -translate-y-1/2 text-muted" />
        <input
          id="global-search"
          ref={inputRef}
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search reviews…"
          className="h-10 w-full rounded-full border border-line bg-card/80 pr-10 pl-10 text-[13px] text-ink placeholder:text-muted hover:border-brand-2/60"
        />
        <kbd aria-hidden="true" className="pointer-events-none absolute top-1/2 right-3.5 hidden -translate-y-1/2 rounded border border-line px-1.5 text-[11px] text-muted sm:block">
          /
        </kbd>
      </form>

      <div className="ml-auto flex items-center gap-2">
        <button
          type="button"
          onClick={() => setAssistantOpen(true)}
          className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-card/70 px-3 py-1.5 text-xs font-medium text-ink-2 hover:border-brand-2/60 hover:text-ink xl:hidden"
        >
          <Sparkles size={14} className="text-teal" aria-hidden="true" /> <span className="hidden xs:inline">Assistant</span>
        </button>
        <ApiStatus />
        <NotificationsButton />
      </div>
    </header>
  )
}
