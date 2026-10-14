import type { ReactNode } from 'react'
import { AssistantPanel } from '../assistant/AssistantPanel'

interface PageLayoutProps {
  children: ReactNode
  /** Extra cards shown above the assistant in the right column on wide screens (e.g. the sentiment donut). */
  rail?: ReactNode
  /** The LLM Assistant page is the assistant, so it hides the side panel. */
  showAssistant?: boolean
}

/**
 * Main column plus the sticky right column. Below 1280px the right column disappears and the
 * assistant becomes a slide-over opened from the top bar, so pages never need two layouts.
 */
export function PageLayout({ children, rail, showAssistant = true }: PageLayoutProps) {
  if (!showAssistant) return <div className="mx-auto max-w-[1280px] space-y-4 @container">{children}</div>
  return (
    <div className="xl:grid xl:grid-cols-[minmax(0,1fr)_336px] xl:items-start xl:gap-4">
      <div className="min-w-0 space-y-4 @container">{children}</div>
      <aside
        aria-label="Filters and assistant"
        className="xl:sticky xl:top-[calc(var(--topbar-h)+16px)] xl:flex xl:h-[calc(100dvh-var(--topbar-h)-32px)] xl:flex-col xl:gap-3 xl:overflow-y-auto"
      >
        {rail && <div className="hidden shrink-0 flex-col gap-3 xl:flex">{rail}</div>}
        <AssistantPanel />
      </aside>
    </div>
  )
}
