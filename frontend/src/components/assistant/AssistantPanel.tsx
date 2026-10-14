import { PanelRightClose, PanelRightOpen, Sparkles, X } from 'lucide-react'
import { useEffect } from 'react'
import { useAssistant } from '@/context/AssistantContext'
import { useUi } from '@/context/UiContext'
import { AssistantChat } from './AssistantChat'

/**
 * Tall assistant column on wide screens (>= 1280px). On smaller screens the same panel slides in
 * from the right and is opened from the "Assistant" button in the top bar.
 */
export function AssistantPanel() {
  const { assistantOpen, setAssistantOpen, assistantCollapsed, setAssistantCollapsed } = useUi()
  const { lastStatus } = useAssistant()

  useEffect(() => {
    if (!assistantOpen) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setAssistantOpen(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [assistantOpen, setAssistantOpen])

  return (
    <>
      {assistantOpen && <div className="fixed inset-0 z-40 bg-black/60 xl:hidden" onClick={() => setAssistantOpen(false)} aria-hidden="true" />}

      {assistantCollapsed && (
        <button
          type="button"
          onClick={() => setAssistantCollapsed(false)}
          className="hidden items-center gap-2 rounded-xl border border-line bg-card px-4 py-3 text-left text-sm font-medium text-ink hover:border-brand-2/60 xl:flex"
        >
          <Sparkles size={16} className="text-teal" aria-hidden="true" />
          Review Assistant
          <PanelRightOpen size={16} className="ml-auto text-muted" aria-hidden="true" />
        </button>
      )}

      <section
        aria-label="Review assistant"
        className={`fixed inset-y-0 right-0 z-50 flex w-[min(420px,100vw)] flex-col overflow-hidden border-l border-line bg-card shadow-2xl transition-[transform,visibility] duration-200 ${
          assistantOpen ? 'translate-x-0' : 'invisible translate-x-full'
        } xl:visible xl:static xl:z-auto xl:min-h-[340px] xl:w-auto xl:flex-1 xl:translate-x-0 xl:rounded-xl xl:border xl:shadow-none ${
          assistantCollapsed ? 'xl:hidden' : ''
        }`}
      >
        <header className="flex items-center gap-2 border-b border-line-soft px-4 py-3">
          <Sparkles size={17} className="text-teal" aria-hidden="true" />
          <h2 className="text-[15px] font-semibold text-ink">Review Assistant</h2>
          {lastStatus && (
            <span
              className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${
                lastStatus === 'success' ? 'border-pos/30 bg-pos/15 text-pos' : 'border-neu/30 bg-neu/15 text-neu'
              }`}
              title={lastStatus === 'success' ? 'Answers are written by the language model from retrieved reviews' : 'The backend is returning retrieved reviews and database numbers without a written answer'}
            >
              {lastStatus === 'success' ? 'LLM on' : 'Search only'}
            </span>
          )}
          <button
            type="button"
            onClick={() => setAssistantCollapsed(true)}
            aria-label="Collapse assistant"
            className="ml-auto hidden rounded-md p-1 text-muted hover:text-ink xl:inline-flex"
          >
            <PanelRightClose size={17} aria-hidden="true" />
          </button>
          <button type="button" onClick={() => setAssistantOpen(false)} aria-label="Close assistant" className="ml-auto rounded-md p-1 text-muted hover:text-ink xl:hidden">
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <AssistantChat variant="panel" />
      </section>
    </>
  )
}
