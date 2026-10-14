import { AlertTriangle, Bell, CheckCircle2, Info, WifiOff } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { useHealth, useMonitoring } from '@/api/queries'
import { useAssistant } from '@/context/AssistantContext'

interface Alert {
  id: string
  level: 'error' | 'warn' | 'info'
  title: string
  detail: ReactNode
  to?: string
}

/** System notices worked out from live API responses. Nothing here is hard-coded. */
export function useAlerts(): Alert[] {
  const health = useHealth()
  const monitoring = useMonitoring()
  const { lastStatus } = useAssistant()
  const alerts: Alert[] = []

  if (health.isError) {
    alerts.push({ id: 'api-down', level: 'error', title: 'API not reachable', detail: 'The dashboard cannot reach the backend, so most panels will show errors.', to: '/monitoring' })
  } else if (health.data && health.data.health.status !== 'healthy') {
    const missing = Object.entries(health.data.health.artifacts_ready)
      .filter(([, ready]) => !ready)
      .map(([name]) => name.replace(/_/g, ' '))
    const nextStep = Object.values(health.data.health.setup_steps ?? {})[0]
    alerts.push({ id: 'degraded', level: 'warn', title: 'Backend is missing files', detail: `Not ready: ${missing.join(', ')}.${nextStep ? ` Next step: ${nextStep}` : ''}`, to: '/monitoring' })
  }

  const retrieval = health.data?.health.retrieval
  if (retrieval && !retrieval.ready) {
    alerts.push({
      id: 'index',
      level: 'warn',
      title: 'Review search index unavailable',
      detail: retrieval.limited_by === 'stale_index' ? 'The index was built from older data. Rebuild it to use the assistant.' : 'Build the index to let the assistant search reviews.',
      to: '/monitoring',
    })
  } else if (retrieval && (retrieval.coverage_pct < 100 || retrieval.limited_by)) {
    alerts.push({
      id: 'index-partial',
      level: 'info',
      title: 'Search covers part of the dataset',
      detail: `${retrieval.indexed_reviews.toLocaleString('en-US')} of ${retrieval.eligible_reviews.toLocaleString('en-US')} reviews are indexed (${retrieval.coverage_pct}%).`,
      to: '/monitoring',
    })
  }

  if (monitoring.data?.status === 'drift_detected') {
    const { drifted_features_count: n, total_features_evaluated: total } = monitoring.data
    alerts.push({ id: 'drift', level: 'warn', title: 'Data drift detected', detail: `${n} of ${total} checked features changed between the reference and current data.`, to: '/monitoring' })
  }

  const llmConfigured = health.data?.health.assistant.llm_configured
  if (llmConfigured === false || (lastStatus && lastStatus !== 'success')) {
    alerts.push({
      id: 'llm-off',
      level: 'info',
      title: 'Assistant is search-only',
      detail:
        llmConfigured === false
          ? 'No Gemini API key is configured on the server, so answers are not written by a language model.'
          : `The last answer was not written by Gemini (${lastStatus?.replace(/_/g, ' ')}).`,
      to: '/assistant',
    })
  }
  return alerts
}

const ICON = { error: WifiOff, warn: AlertTriangle, info: Info } as const
const TONE = { error: 'text-neg', warn: 'text-neu', info: 'text-blue' } as const

export function NotificationsButton() {
  const alerts = useAlerts()
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onPointer = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', onPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onPointer)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="true"
        aria-label={alerts.length ? `Notifications, ${alerts.length} active` : 'Notifications, none active'}
        className="relative flex size-9 items-center justify-center rounded-lg text-ink-2 hover:bg-card-2 hover:text-ink"
      >
        <Bell size={18} aria-hidden="true" />
        {alerts.length > 0 && (
          <span aria-hidden="true" className="absolute top-0.5 right-0.5 flex size-4 items-center justify-center rounded-full bg-neg text-[10px] font-semibold text-white">
            {alerts.length}
          </span>
        )}
      </button>
      {open && (
        <div role="region" aria-label="Notifications" className="absolute top-11 right-0 z-50 w-[min(340px,calc(100vw-2rem))] rounded-xl border border-line bg-card p-2 shadow-2xl">
          <p className="px-2 py-1.5 text-xs font-semibold text-ink">System notices</p>
          {alerts.length === 0 ? (
            <p className="flex items-center gap-2 px-2 py-3 text-[13px] text-muted">
              <CheckCircle2 size={16} className="text-pos" aria-hidden="true" /> Everything looks fine right now.
            </p>
          ) : (
            <ul className="space-y-1">
              {alerts.map((a) => {
                const Icon = ICON[a.level]
                return (
                  <li key={a.id} className="flex gap-2.5 rounded-lg p-2 hover:bg-card-2/70">
                    <Icon size={16} className={`mt-0.5 shrink-0 ${TONE[a.level]}`} aria-hidden="true" />
                    <div className="min-w-0 text-[13px]">
                      <p className="font-medium text-ink">{a.title}</p>
                      <p className="mt-0.5 text-xs leading-relaxed text-muted">{a.detail}</p>
                      {a.to && (
                        <Link to={a.to} onClick={() => setOpen(false)} className="mt-1 inline-block text-xs font-medium text-brand-2 hover:underline">
                          Open details
                        </Link>
                      )}
                    </div>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
