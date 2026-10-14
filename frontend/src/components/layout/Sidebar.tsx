import { Activity, Bot, Brain, Home, MessageSquareText, Network, ShieldCheck, Shapes, SmilePlus, Sparkles, X } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { useEffect } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { useHealth } from '@/api/queries'
import { useUi } from '@/context/UiContext'

interface NavItem {
  to: string
  label: string
  icon: LucideIcon
}

export const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Overview', icon: Home },
  { to: '/reviews', label: 'Reviews Analysis', icon: MessageSquareText },
  { to: '/sentiment', label: 'Sentiment Analysis', icon: SmilePlus },
  { to: '/topics', label: 'Topic Modeling', icon: Shapes },
  { to: '/predictions', label: 'ML Predictions', icon: Brain },
  { to: '/deep-learning', label: 'Deep Learning (NLP)', icon: Network },
  { to: '/assistant', label: 'LLM Assistant', icon: Bot },
  { to: '/monitoring', label: 'Deploy & Monitor', icon: ShieldCheck },
]

function DatasetCard() {
  const { data, isError, isPending } = useHealth()
  const ready = data?.health.artifacts_ready
  const loaded = ready?.parquet_data === true
  const state = isPending ? { text: 'Checking…', dot: 'bg-muted' } : isError ? { text: 'API offline', dot: 'bg-neg' } : loaded ? { text: 'Loaded', dot: 'bg-pos' } : { text: 'Not loaded', dot: 'bg-neu' }
  return (
    <div className="rounded-xl border border-line bg-card/80 p-3.5">
      <p className="text-xs text-muted">Dataset</p>
      <p className="mt-0.5 text-[13px] font-medium text-ink">Amazon Reviews 2023</p>
      <p className="mt-1.5 flex items-center gap-1.5 text-xs text-ink-2">
        <span aria-hidden="true" className={`size-2 rounded-full ${state.dot}`} />
        {state.text}
      </p>
    </div>
  )
}

/** Fixed navigation on wide screens, slide-in drawer below 1024px. */
export function Sidebar() {
  const { navOpen, setNavOpen } = useUi()
  const { pathname } = useLocation()

  // Close the drawer after navigating so it does not cover the new page.
  useEffect(() => {
    setNavOpen(false)
  }, [pathname, setNavOpen])
  useEffect(() => {
    if (!navOpen) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setNavOpen(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [navOpen, setNavOpen])

  return (
    <>
      {navOpen && <div className="fixed inset-0 z-40 bg-black/60 lg:hidden" onClick={() => setNavOpen(false)} aria-hidden="true" />}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-[222px] flex-col overflow-y-auto border-r border-line bg-sidebar transition-[transform,visibility] duration-200 lg:visible lg:translate-x-0 ${
          navOpen ? 'translate-x-0' : 'invisible -translate-x-full'
        }`}
      >
        <div className="flex h-[var(--topbar-h)] items-center gap-2.5 px-4">
          <Link to="/" className="flex items-center gap-2.5" aria-label="ReviewRadar home">
            <span aria-hidden="true" className="flex size-9 items-center justify-center rounded-lg bg-gradient-to-br from-brand to-teal text-white">
              <Activity size={20} strokeWidth={2.5} />
            </span>
            <span className="leading-tight">
              <span className="block text-[17px] font-semibold tracking-tight text-ink">ReviewRadar</span>
              <span className="block text-[11px] text-muted">Amazon Review Intelligence</span>
            </span>
          </Link>
          <button type="button" onClick={() => setNavOpen(false)} aria-label="Close menu" className="ml-auto rounded-md p-1 text-muted hover:text-ink lg:hidden">
            <X size={18} aria-hidden="true" />
          </button>
        </div>

        <nav aria-label="Main" className="mt-2 flex-1 space-y-1 px-3">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2.5 text-[13.5px] font-medium transition-colors ${
                  isActive ? 'bg-brand text-white shadow-[0_6px_18px_-8px_rgb(63_99_240/0.9)]' : 'text-ink-2 hover:bg-card-2/70 hover:text-ink'
                }`
              }
            >
              <Icon size={18} aria-hidden="true" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="space-y-3 p-3">
          <div className="rounded-xl border border-line bg-card/80 p-3.5">
            <Sparkles size={16} className="text-ink" aria-hidden="true" />
            <p className="mt-2 text-[13px] leading-snug font-medium text-ink">Find common customer complaints</p>
            <p className="mt-1 text-xs leading-relaxed text-muted">Browse real reviews, test the sentiment model, or ask the assistant a question.</p>
            <Link to="/predictions" className="mt-2.5 inline-block text-xs font-medium text-brand-2 hover:underline">
              Try the sentiment tester
            </Link>
          </div>
          <DatasetCard />
        </div>
      </aside>
    </>
  )
}
